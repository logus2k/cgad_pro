/**
 * MeshLoader - Client-side HDF5 mesh loader using h5wasm
 * 
 * Loads mesh files directly in the browser, enabling:
 * - Early mesh loading (on gallery selection, before solve)
 * - Geometry ready before solver starts
 * - Progressive solution updates visible immediately
 */

// Import h5wasm from CDN
let h5wasm = null;
let h5wasmReady = null;

/**
 * Initialize h5wasm library (lazy load)
 */
async function initH5Wasm() {
    if (h5wasmReady) return h5wasmReady;
    
    h5wasmReady = (async () => {
        console.log('Loading h5wasm library...');
        const module = await import('../library/hdf5_hl.js');
        h5wasm = module.default || module;
        await h5wasm.ready;
        console.log('h5wasm ready');
        return h5wasm;
    })();
    
    return h5wasmReady;
}

export class MeshLoader {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.cache = new Map();  // Cache loaded meshes by URL
        this.loading = new Map(); // Track in-progress loads
        this.onProgress = null;  // Progress callback
    }
    
    /**
     * Set progress callback
     * @param {Function} callback - (stage, progress) => void
     */
    setProgressCallback(callback) {
        this.onProgress = callback;
    }
    
    /**
     * Report progress
     */
    _reportProgress(stage, progress = null) {
        if (this.onProgress) {
            this.onProgress(stage, progress);
        }
    }
    
    /**
     * Load mesh from HDF5 file
     * @param {string} url - URL to .h5 file (e.g., '/mesh/y_tube1_772k.h5')
     * @param {Object} options - Loading options
     * @returns {Promise<Object>} - { coordinates: {x, y}, connectivity, nodes, elements }
     */
    async load(url, options = {}) {
        const fullUrl = this.baseUrl + url;
        
        // Return cached result if available
        if (this.cache.has(fullUrl)) {
            console.log(`Mesh cache hit: ${url}`);
            return this.cache.get(fullUrl);
        }
        
        // Return existing promise if already loading
        if (this.loading.has(fullUrl)) {
            console.log(`Mesh already loading: ${url}`);
            return this.loading.get(fullUrl);
        }
        
        // Start new load
        const loadPromise = this._loadMesh(fullUrl, options);
        this.loading.set(fullUrl, loadPromise);
        
        try {
            const result = await loadPromise;
            this.cache.set(fullUrl, result);
            return result;
        } finally {
            this.loading.delete(fullUrl);
        }
    }
    
    /**
     * Preload mesh (non-blocking, for gallery selection)
     * @param {string} url - URL to .h5 file
     * @returns {Promise<Object>} - Same as load()
     */
    preload(url) {
        console.log(`Preloading mesh: ${url}`);
        return this.load(url);
    }
    
    /**
     * Check if mesh is cached
     * @param {string} url - URL to check
     * @returns {boolean}
     */
    isCached(url) {
        return this.cache.has(this.baseUrl + url);
    }
    
    /**
     * Check if mesh is currently loading
     * @param {string} url - URL to check
     * @returns {boolean}
     */
    isLoading(url) {
        return this.loading.has(this.baseUrl + url);
    }
    
    /**
     * Get cached mesh (synchronous, returns null if not cached)
     * @param {string} url - URL to get
     * @returns {Object|null}
     */
    getCached(url) {
        return this.cache.get(this.baseUrl + url) || null;
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }
    
    /**
     * Internal: Load mesh from HDF5 file
     */
    async _loadMesh(url, options) {
        const startTime = performance.now();
        
        try {
            // Initialize h5wasm if needed
            this._reportProgress('init_h5wasm');
            await initH5Wasm();
            
            // Fetch the HDF5 file
            this._reportProgress('downloading');
            console.log(`Fetching mesh: ${url}`);
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch mesh: ${response.status} ${response.statusText}`);
            }
            
            const contentLength = response.headers.get('content-length');
            let arrayBuffer;
            
            if (contentLength && this.onProgress) {
                // Stream with progress reporting
                arrayBuffer = await this._fetchWithProgress(response, parseInt(contentLength));
            } else {
                arrayBuffer = await response.arrayBuffer();
            }
            
            const downloadTime = performance.now() - startTime;
            console.log(`Downloaded ${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB in ${downloadTime.toFixed(0)}ms`);
            
            // Parse HDF5 file
            this._reportProgress('parsing');
            const parseStart = performance.now();
            
            const meshData = await this._parseH5File(arrayBuffer, url);
            
            const parseTime = performance.now() - parseStart;
            const totalTime = performance.now() - startTime;
            
            console.log(`Parsed mesh in ${parseTime.toFixed(0)}ms (total: ${totalTime.toFixed(0)}ms)`);
            console.log(`   Nodes: ${meshData.nodes}, Elements: ${meshData.elements}`);
            
            this._reportProgress('complete');
            
            return meshData;
            
        } catch (error) {
            this._reportProgress('error');
            console.error('Mesh loading failed:', error);
            throw error;
        }
    }
    
    /**
     * Fetch with progress reporting
     */
    async _fetchWithProgress(response, contentLength) {
        const reader = response.body.getReader();
        const chunks = [];
        let receivedLength = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            chunks.push(value);
            receivedLength += value.length;
            
            const progress = receivedLength / contentLength;
            this._reportProgress('downloading', progress);
        }
        
        // Combine chunks into single ArrayBuffer
        const arrayBuffer = new ArrayBuffer(receivedLength);
        const uint8Array = new Uint8Array(arrayBuffer);
        let position = 0;
        
        for (const chunk of chunks) {
            uint8Array.set(chunk, position);
            position += chunk.length;
        }
        
        return arrayBuffer;
    }
    
    /**
     * Parse HDF5 file using h5wasm
     */
    async _parseH5File(arrayBuffer, filename) {
        const { FS } = await h5wasm.ready;
        
        // Write file to virtual filesystem
        const tempFilename = 'temp_mesh_' + Date.now() + '.h5';
        FS.writeFile(tempFilename, new Uint8Array(arrayBuffer));
        
        try {
            // Open HDF5 file
            const file = new h5wasm.File(tempFilename, 'r');
            
            // Read datasets
            // Expected structure: /x, /y, /quad8 (based on your mesh files)
            const xDataset = file.get('x');
            const yDataset = file.get('y');
            const quad8Dataset = file.get('quad8');
            
            if (!xDataset || !yDataset || !quad8Dataset) {
                // Try alternate structure
                const keys = file.keys();
                console.log('HDF5 keys:', keys);
                throw new Error(`Missing required datasets. Found: ${keys.join(', ')}`);
            }
            
            // Extract data
            const x = xDataset.value;  // Float64Array or Float32Array
            const y = yDataset.value;
            const quad8Raw = quad8Dataset.value;  // Int32Array or similar
            
            // Get shape info for connectivity
            const quad8Shape = quad8Dataset.shape;  // [num_elements, 8]
            const numElements = quad8Shape[0];
            const nodesPerElement = quad8Shape[1];
            
            console.log(`   x: ${x.length} values, dtype: ${xDataset.dtype}`);
            console.log(`   y: ${y.length} values, dtype: ${yDataset.dtype}`);
            console.log(`   quad8: ${numElements} elements x ${nodesPerElement} nodes`);
            
            // Convert to regular arrays for compatibility
            const xArray = Array.from(x);
            const yArray = Array.from(y);
            
            // Reshape connectivity to 2D array
            const connectivity = [];
            for (let i = 0; i < numElements; i++) {
                const element = [];
                for (let j = 0; j < nodesPerElement; j++) {
                    element.push(quad8Raw[i * nodesPerElement + j]);
                }
                connectivity.push(element);
            }
            
            // Close file
            file.close();
            
            return {
                coordinates: { x: xArray, y: yArray },
                connectivity: connectivity,
                nodes: x.length,
                elements: numElements
            };
            
        } finally {
            // Clean up temp file
            try {
                FS.unlink(tempFilename);
            } catch (e) {
                // Ignore cleanup errors
            }
        }
    }
}

// Create singleton instance
export const meshLoader = new MeshLoader();

// Also export for direct use
export default MeshLoader;
