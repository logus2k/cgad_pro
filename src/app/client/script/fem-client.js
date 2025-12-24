/**
 * FEM Solver Client - Socket.IO connection manager
 */
export class FEMClient {

    constructor(serverUrl = window.location.origin, basePath = '/fem') {

        this.serverUrl = serverUrl;
        this.basePath = basePath;
        
        const socketOrigin = new URL(serverUrl).origin;
        this.socket = io(socketOrigin, {
            path: `${basePath}/socket.io`,
        });
        this.currentJobId = null;
        this.eventHandlers = {};
        
        this.setupConnectionHandlers();
    }
    
    setupConnectionHandlers() {

        this.socket.on('connect', () => {
            console.log('Connected to FEM server');
            this.triggerEvent('connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from FEM server');
            this.triggerEvent('disconnected');
        });
        
        // Stage events
        this.socket.on('stage_start', (data) => {
            console.log(`Stage started: ${data.stage}`);
            this.triggerEvent('stage_start', data);
        });
        
        this.socket.on('stage_complete', (data) => {
            console.log(`Stage complete: ${data.stage} (${data.duration.toFixed(2)}s)`);
            this.triggerEvent('stage_complete', data);
        });
        
        // Mesh loaded - fetch binary
        this.socket.on('mesh_loaded', async (data) => {
            console.log(`Mesh loaded: ${data.nodes} nodes, ${data.elements} elements`);
            
            // Fetch binary mesh data
            if (data.binary_url) {
                try {
                    const meshData = await this.fetchBinaryMesh(data.binary_url);
                    data.coordinates = meshData.coordinates;
                    data.connectivity = meshData.connectivity;
                } catch (error) {
                    console.error('Failed to fetch binary mesh:', error);
                }
            }
            
            this.triggerEvent('mesh_loaded', data);
        });
        
        // Solving progress
        this.socket.on('solve_progress', (data) => {
            this.triggerEvent('solve_progress', data);
        });
        
        // Incremental solution update
        this.socket.on('solution_update', (data) => {
            // Decode base64 binary chunk
            const binaryData = this.base64ToArrayBuffer(data.chunk_data);
            data.binary_chunk = new Float32Array(binaryData);
            this.triggerEvent('solution_update', data);
        });

        // Incremental solution update
        this.socket.on('solution_increment', (data) => {
            // Decode base64 binary chunk
            const binaryData = this.base64ToArrayBuffer(data.chunk_data);
            const solutionChunk = new Float32Array(binaryData);
            
            // Reconstruct full solution from subsampled data
            const fullSolution = this.reconstructSolution(
                solutionChunk,
                data.chunk_info
            );
            
            data.solution_values = fullSolution;
            this.triggerEvent('solution_increment', data);
        });        
        
        // Solve complete
        this.socket.on('solve_complete', async (data) => {
            console.log(`Solve complete! Converged: ${data.converged}, Iterations: ${data.iterations}`);
            
            // Fetch final solution binary
            if (data.solution_url) {
                try {
                    const solution = await this.fetchBinarySolution(data.solution_url);
                    data.solution_field = solution;
                } catch (error) {
                    console.error('Failed to fetch solution:', error);
                }
            }
            
            this.triggerEvent('solve_complete', data);
        });
        
        // Error
        this.socket.on('solve_error', (data) => {
            console.error(`Solver error at ${data.stage}:`, data.error);
            this.triggerEvent('solve_error', data);
        });
    }
    
    /**
     * Fetch binary mesh data
     */
    async fetchBinaryMesh(url) {
        // url already includes basePath from server
        const response = await fetch(`${this.serverUrl}${url}`);
        const buffer = await response.arrayBuffer();
        
        return this.parseBinaryMesh(buffer);
    }
    
    /**
     * Parse binary mesh format
     */
    parseBinaryMesh(buffer) {
        const view = new DataView(buffer);
        let offset = 0;
        
        // Read header
        const numNodes = view.getUint32(offset, true); offset += 4;
        const numElements = view.getUint32(offset, true); offset += 4;
        
        console.log(`Parsing binary mesh: ${numNodes} nodes, ${numElements} elements`);
        
        // Read coordinates
        const x = new Float32Array(buffer, offset, numNodes);
        offset += numNodes * 4;
        
        const y = new Float32Array(buffer, offset, numNodes);
        offset += numNodes * 4;
        
        // Read connectivity (8 nodes per Quad-8 element)
        const connectivityFlat = new Int32Array(buffer, offset, numElements * 8);
        
        // Reshape to 2D array
        const connectivity = [];
        for (let i = 0; i < numElements; i++) {
            connectivity.push(
                Array.from(connectivityFlat.slice(i * 8, (i + 1) * 8))
            );
        }
        
        return {
            coordinates: { x: Array.from(x), y: Array.from(y) },
            connectivity: connectivity
        };
    }

    /**
     * Reconstruct full solution from subsampled chunk
     */
    reconstructSolution(chunk, info) {
        const { stride, total_nodes } = info;
        
        // Create full array with interpolated values
        const full = new Float32Array(total_nodes);
        
        for (let i = 0; i < chunk.length - 1; i++) {
            const baseIndex = i * stride;
            const value1 = chunk[i];
            const value2 = chunk[i + 1];
            
            // Set sampled value
            full[baseIndex] = value1;
            
            // Linearly interpolate between samples
            for (let j = 1; j < stride && baseIndex + j < total_nodes; j++) {
                const t = j / stride;
                full[baseIndex + j] = value1 * (1 - t) + value2 * t;
            }
        }
        
        // Handle last value
        if (chunk.length > 0) {
            full[(chunk.length - 1) * stride] = chunk[chunk.length - 1];
        }
        
        return full;
    }    
    
    /**
     * Fetch binary solution data
     */
    async fetchBinarySolution(url) {
        // url already includes basePath from server
        const response = await fetch(`${this.serverUrl}${url}`);
        const buffer = await response.arrayBuffer();
        
        const view = new DataView(buffer);
        const numValues = view.getUint32(0, true);
        
        return new Float32Array(buffer, 4, numValues);
    }
    
    /**
     * Convert base64 to ArrayBuffer
     */
    base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    /**
     * Start a new solve job
     */
    async startSolve(params) {
        const response = await fetch(`${this.serverUrl}${this.basePath}/solve`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        this.currentJobId = data.job_id;

        const join = () => {
            this.socket.emit('join_room', { job_id: data.job_id });
            console.log(`Joined room ${data.job_id}`);
        };

        if (this.socket.connected) {
            join();
        } else {
            this.socket.once('connect', join);
        }

        console.log(`Job started: ${data.job_id}`);
        this.triggerEvent('job_started', data);

        return data;
    }
    
    /**
     * Get current job status
     */
    async getJobStatus(jobId) {
        const response = await fetch(`${this.serverUrl}${this.basePath}/solve/${jobId}/status`);
        return response.json();
    }
    
    /**
     * Get job results
     */
    async getJobResults(jobId) {
        const response = await fetch(`${this.serverUrl}${this.basePath}/solve/${jobId}/results`);
        return response.json();
    }
    
    /**
     * Get available meshes
     */
    async getMeshes() {
        const response = await fetch(`${this.serverUrl}${this.basePath}/meshes`);
        return response.json();
    }
    
    /**
     * Register event handler
     */
    on(eventName, handler) {
        if (!this.eventHandlers[eventName]) {
            this.eventHandlers[eventName] = [];
        }
        this.eventHandlers[eventName].push(handler);
    }
    
    /**
     * Trigger event handlers
     */
    triggerEvent(eventName, data) {
        if (this.eventHandlers[eventName]) {
            this.eventHandlers[eventName].forEach(handler => handler(data));
        }
    }
}
