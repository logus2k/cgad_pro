/**
 * ProfilingHDF5Loader - Loads profiling timeline from HDF5 with progress.
 * 
 * Uses h5wasm to read HDF5 files in browser.
 * Returns typed arrays ready for InstancedMesh rendering.
 * 
 * Features:
 * - Accurate download progress via Content-Length
 * - Pre-grouped/sorted data (no client-side processing)
 * - Typed arrays for direct GPU upload
 * - Deduplicated names with index lookup
 * 
 * Location: /src/app/client/script/profiling/profiling_hdf5_loader.js
 */

// h5wasm module (loaded once)
let h5wasmReady = null;
let h5wasm = null;

/**
 * Initialize h5wasm library.
 * Call once at app startup or lazily on first use.
 * 
 * Uses same path as mesh-loader.js: ../library/hdf5_hl.js
 * This file is in /src/app/client/script/profiling/
 * So path is ../../library/hdf5_hl.js
 */
export async function initH5Wasm() {
    if (h5wasmReady) {
        return h5wasmReady;
    }
    
    h5wasmReady = (async () => {
        console.log('[ProfilingHDF5Loader] Loading h5wasm library...');
        // Path relative to /src/app/client/script/profiling/
        const module = await import('../../library/hdf5_hl.js');
        h5wasm = module.default || module;
        await h5wasm.ready;
        console.log('[ProfilingHDF5Loader] h5wasm ready');
        return h5wasm;
    })();
    
    return h5wasmReady;
}

/**
 * Load profiling timeline from HDF5 URL.
 * 
 * @param {string} url - URL to HDF5 file
 * @param {Function} onProgress - Progress callback: (loaded, total, phase) => void
 *   - phase: 'download' | 'parse'
 *   - loaded/total in bytes for download, percentage for parse
 * @returns {Promise<ProfilingData>}
 */
export async function loadProfilingHDF5(url, onProgress = null) {
    // Ensure h5wasm is ready
    await initH5Wasm();
    
    // Phase 1: Download with progress
    const buffer = await fetchWithProgress(url, (loaded, total) => {
        if (onProgress) onProgress(loaded, total, 'download');
    });
    
    if (onProgress) onProgress(100, 100, 'parse');
    
    // Phase 2: Parse HDF5
    const data = await parseHDF5(buffer);
    
    return data;
}

/**
 * Fetch URL as ArrayBuffer with progress.
 */
async function fetchWithProgress(url, onProgress) {
    const response = await fetch(url);
    
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    
    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (!response.body) {
        // Fallback for browsers without streaming
        const buffer = await response.arrayBuffer();
        if (onProgress) onProgress(buffer.byteLength, buffer.byteLength);
        return buffer;
    }
    
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    
    while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        chunks.push(value);
        loaded += value.length;
        
        if (onProgress) {
            onProgress(loaded, total || loaded);
        }
    }
    
    // Combine chunks into single buffer
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }
    
    return buffer.buffer;
}

/**
 * Parse HDF5 buffer into ProfilingData structure.
 * 
 * Uses Emscripten filesystem (same pattern as mesh-loader.js)
 * 
 * @param {ArrayBuffer} buffer 
 * @returns {ProfilingData}
 */
async function parseHDF5(buffer) {
    // Get FS from h5wasm.ready (same as mesh-loader.js)
    const { FS } = await h5wasm.ready;
    
    const tempFilename = 'temp_timeline_' + Date.now() + '.h5';
    
    // Write buffer to virtual filesystem
    FS.writeFile(tempFilename, new Uint8Array(buffer));
    
    try {
        // Open file from virtual filesystem
        const file = new h5wasm.File(tempFilename, 'r');
        
        try {
            return parseHDF5FromFile(file);
        } finally {
            file.close();
        }
    } finally {
        // Clean up temp file
        try {
            FS.unlink(tempFilename);
        } catch (e) {
            // Ignore cleanup errors
        }
    }
}

/**
 * Parse HDF5 from opened file handle
 */
function parseHDF5FromFile(file) {
    // Read metadata
    const meta = file.get('meta');
    const sessionId = meta.attrs['session_id']?.value || 'unknown';
    const totalDurationNs = meta.attrs['total_duration_ns']?.value || 0n;
    const totalEvents = meta.attrs['total_events']?.value || 0;
    
    // Read categories present
    const categoriesDataset = meta.get('categories');
    const categories = categoriesDataset ? Array.from(categoriesDataset.value) : [];
    
    // Read names lookup
    const namesGroup = file.get('names');
    const names = {};
    if (namesGroup) {
        for (const category of categories) {
            const namesDataset = namesGroup.get(category);
            if (namesDataset) {
                names[category] = Array.from(namesDataset.value);
            }
        }
    }
    
    // Read each category's data
    const categoryData = {};
    
    for (const category of categories) {
        const catGroup = file.get(category);
        if (!catGroup) continue;
        
        const data = {
            // Core fields (typed arrays)
            start_ns: getDataset(catGroup, 'start_ns'),      // BigUint64Array
            duration_ns: getDataset(catGroup, 'duration_ns'), // Uint32Array
            stream: getDataset(catGroup, 'stream'),           // Uint8Array
            name_idx: getDataset(catGroup, 'name_idx'),       // Uint16Array
            
            // Names lookup
            names: names[category] || [],
            
            // Count
            count: 0
        };
        
        data.count = data.start_ns?.length || 0;
        
        // Category-specific fields
        if (category === 'cuda_kernel') {
            data.grid_x = getDataset(catGroup, 'grid_x');
            data.grid_y = getDataset(catGroup, 'grid_y');
            data.grid_z = getDataset(catGroup, 'grid_z');
            data.block_x = getDataset(catGroup, 'block_x');
            data.block_y = getDataset(catGroup, 'block_y');
            data.block_z = getDataset(catGroup, 'block_z');
            data.registers = getDataset(catGroup, 'registers');
            data.shared_static = getDataset(catGroup, 'shared_static');
            data.shared_dynamic = getDataset(catGroup, 'shared_dynamic');
        } else if (category.startsWith('cuda_memcpy')) {
            data.bytes = getDataset(catGroup, 'bytes');
        } else if (category === 'nvtx_range') {
            data.color = getDataset(catGroup, 'color');
        }
        
        categoryData[category] = data;
    }
    
    return {
        sessionId,
        totalDurationNs: Number(totalDurationNs),
        totalEvents,
        categories,
        categoryData,
        
        // Helper method: convert to flat events array (for compatibility)
        toEvents() {
            return convertToEvents(this);
        },
        
        // Helper method: get typed arrays for renderer
        getRendererData() {
            return prepareRendererData(this);
        }
    };
}

/**
 * Get dataset value as typed array.
 */
function getDataset(group, name) {
    const dataset = group.get(name);
    if (!dataset) return null;
    
    const value = dataset.value;
    
    // h5wasm returns typed arrays directly
    return value;
}

/**
 * Convert HDF5 data to flat events array (for compatibility with existing code).
 * 
 * Note: This reconstructs the original event format. For best performance,
 * use getRendererData() instead which keeps typed arrays.
 */
function convertToEvents(data) {
    const events = [];
    let eventId = 0;
    
    for (const category of data.categories) {
        const catData = data.categoryData[category];
        if (!catData || catData.count === 0) continue;
        
        const names = catData.names;
        
        for (let i = 0; i < catData.count; i++) {
            const startNs = Number(catData.start_ns[i]);
            const durationNs = catData.duration_ns[i];
            
            const event = {
                id: eventId++,
                category,
                name: names[catData.name_idx[i]] || '',
                start_ns: startNs,
                end_ns: startNs + durationNs,
                duration_ns: durationNs,
                stream: catData.stream[i],
                metadata: {}
            };
            
            // Category-specific metadata
            if (category === 'cuda_kernel') {
                event.metadata.grid = [
                    catData.grid_x?.[i] || 1,
                    catData.grid_y?.[i] || 1,
                    catData.grid_z?.[i] || 1
                ];
                event.metadata.block = [
                    catData.block_x?.[i] || 1,
                    catData.block_y?.[i] || 1,
                    catData.block_z?.[i] || 1
                ];
                event.metadata.registers_per_thread = catData.registers?.[i] || 0;
                event.metadata.shared_memory_static = catData.shared_static?.[i] || 0;
                event.metadata.shared_memory_dynamic = catData.shared_dynamic?.[i] || 0;
            } else if (category.startsWith('cuda_memcpy')) {
                event.metadata.bytes = catData.bytes?.[i] || 0;
            } else if (category === 'nvtx_range') {
                event.metadata.color = catData.color?.[i] || 0;
            }
            
            events.push(event);
        }
    }
    
    return events;
}

/**
 * Prepare data optimized for TimelineRenderer.
 * 
 * Returns typed arrays that can be directly used for InstancedMesh:
 * - Per-category arrays already sorted by start time
 * - Float32 positions in milliseconds (ready for GPU)
 */
function prepareRendererData(data) {
    const rendererData = {
        categories: data.categories,
        totalDurationMs: data.totalDurationNs / 1e6,
        totalEvents: data.totalEvents,
        byCategory: {}
    };
    
    for (const category of data.categories) {
        const catData = data.categoryData[category];
        if (!catData || catData.count === 0) continue;
        
        const count = catData.count;
        
        // Convert start_ns (BigUint64) to start_ms (Float32)
        const startMs = new Float32Array(count);
        const durationMs = new Float32Array(count);
        
        for (let i = 0; i < count; i++) {
            startMs[i] = Number(catData.start_ns[i]) / 1e6;
            durationMs[i] = catData.duration_ns[i] / 1e6;
        }
        
        rendererData.byCategory[category] = {
            count,
            startMs,        // Float32Array - ready for GPU
            durationMs,     // Float32Array - ready for GPU
            stream: catData.stream,  // Uint8Array
            nameIdx: catData.name_idx,
            names: catData.names,
            
            // Category-specific (for tooltips)
            metadata: {
                grid_x: catData.grid_x,
                grid_y: catData.grid_y,
                grid_z: catData.grid_z,
                block_x: catData.block_x,
                block_y: catData.block_y,
                block_z: catData.block_z,
                registers: catData.registers,
                shared_static: catData.shared_static,
                shared_dynamic: catData.shared_dynamic,
                bytes: catData.bytes,
                color: catData.color
            }
        };
    }
    
    return rendererData;
}

/**
 * @typedef {Object} ProfilingData
 * @property {string} sessionId
 * @property {number} totalDurationNs
 * @property {number} totalEvents
 * @property {string[]} categories
 * @property {Object.<string, CategoryData>} categoryData
 */

/**
 * @typedef {Object} CategoryData
 * @property {BigUint64Array} start_ns
 * @property {Uint32Array} duration_ns
 * @property {Uint8Array} stream
 * @property {Uint16Array} name_idx
 * @property {string[]} names
 * @property {number} count
 */
