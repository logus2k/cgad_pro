/**
 * FEM Solver Client - Socket.IO connection manager
 */
export class FEMClient {
    constructor(serverUrl = 'http://localhost:4567') {
        this.serverUrl = serverUrl;
        this.socket = io(serverUrl);
        this.currentJobId = null;
        this.eventHandlers = {};
        
        this.setupConnectionHandlers();
    }
    
    setupConnectionHandlers() {
        this.socket.on('connect', () => {
            console.log('✅ Connected to FEM server');
            this.triggerEvent('connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from FEM server');
            this.triggerEvent('disconnected');
        });
        
        // Stage events
        this.socket.on('stage_start', (data) => {
            console.log(`📍 Stage started: ${data.stage}`);
            this.triggerEvent('stage_start', data);
        });
        
        this.socket.on('stage_complete', (data) => {
            console.log(`✅ Stage complete: ${data.stage} (${data.duration.toFixed(2)}s)`);
            this.triggerEvent('stage_complete', data);
        });
        
        // Mesh loaded
        this.socket.on('mesh_loaded', (data) => {
            console.log(`📐 Mesh loaded: ${data.nodes} nodes, ${data.elements} elements`);
            this.triggerEvent('mesh_loaded', data);
        });
        
        // Solving progress
        this.socket.on('solve_progress', (data) => {
            this.triggerEvent('solve_progress', data);
        });
        
        // Solve complete
        this.socket.on('solve_complete', (data) => {
            console.log(`🎉 Solve complete! Converged: ${data.converged}, Iterations: ${data.iterations}`);
            this.triggerEvent('solve_complete', data);
        });
        
        // Error
        this.socket.on('solve_error', (data) => {
            console.error(`❌ Solver error at ${data.stage}:`, data.error);
            this.triggerEvent('solve_error', data);
        });
    }
    
    /**
     * Start a new solve job
     */
    async startSolve(params) {
        try {
            const response = await fetch(`${this.serverUrl}/solve`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.currentJobId = data.job_id;
            
            // Join Socket.IO room for this job
            this.socket.emit('join_room', {job_id: data.job_id});
            
            console.log(`🚀 Job started: ${data.job_id}`);
            this.triggerEvent('job_started', data);
            
            return data;
            
        } catch (error) {
            console.error('Failed to start solve:', error);
            this.triggerEvent('job_error', {error: error.message});
            throw error;
        }
    }
    
    /**
     * Get current job status
     */
    async getJobStatus(jobId) {
        const response = await fetch(`${this.serverUrl}/solve/${jobId}/status`);
        return response.json();
    }
    
    /**
     * Get job results
     */
    async getJobResults(jobId) {
        const response = await fetch(`${this.serverUrl}/solve/${jobId}/results`);
        return response.json();
    }
    
    /**
     * Get available meshes
     */
    async getMeshes() {
        const response = await fetch(`${this.serverUrl}/meshes`);
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
