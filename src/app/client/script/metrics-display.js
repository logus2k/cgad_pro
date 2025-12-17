/**
 * Metrics Display Manager - Updates HUD with real-time solver metrics
 */
export class MetricsDisplay {
    constructor(hudElement) {
        this.container = hudElement.querySelector('.metrics-container');
        this.metrics = {};
        this.createMetricElements();
    }
    
    createMetricElements() {
        this.container.innerHTML = `
            <div class="metric-row">
                <span class="metric-label">Status:</span>
                <span class="metric-value" id="metric-status">Idle</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Current Stage:</span>
                <span class="metric-value" id="metric-stage">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Nodes:</span>
                <span class="metric-value" id="metric-nodes">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Elements:</span>
                <span class="metric-value" id="metric-elements">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Progress:</span>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar"></div>
                    <span class="progress-text" id="progress-text">0%</span>
                </div>
            </div>
            <div class="metric-row">
                <span class="metric-label">Iteration:</span>
                <span class="metric-value" id="metric-iteration">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Residual:</span>
                <span class="metric-value" id="metric-residual">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">ETR:</span>
                <span class="metric-value" id="metric-etr">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Total Time:</span>
                <span class="metric-value" id="metric-total-time">-</span>
            </div>
        `;
    }
    
    updateStatus(status) {
        document.getElementById('metric-status').textContent = status;
    }
    
    updateStage(stage) {
        const stageNames = {
            'load_mesh': 'Loading Mesh',
            'assemble_system': 'Assembling System',
            'apply_bc': 'Applying BC',
            'solve_system': 'Solving',
            'compute_derived': 'Post-Processing'
        };
        document.getElementById('metric-stage').textContent = stageNames[stage] || stage;
    }
    
    updateMesh(nodes, elements) {
        document.getElementById('metric-nodes').textContent = nodes.toLocaleString();
        document.getElementById('metric-elements').textContent = elements.toLocaleString();
    }
    
    updateProgress(iteration, maxIterations, residual, etr) {
        const percent = (iteration / maxIterations * 100).toFixed(1);
        
        document.getElementById('progress-bar').style.width = `${percent}%`;
        document.getElementById('progress-text').textContent = `${percent}%`;
        document.getElementById('metric-iteration').textContent = 
            `${iteration} / ${maxIterations}`;
        document.getElementById('metric-residual').textContent = residual.toExponential(3);
        document.getElementById('metric-etr').textContent = this.formatTime(etr);
    }
    
    updateTotalTime(seconds) {
        document.getElementById('metric-total-time').textContent = 
            this.formatTime(seconds);
    }
    
    formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    }
    
    reset() {
        this.updateStatus('Idle');
        document.getElementById('metric-stage').textContent = '-';
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('metric-iteration').textContent = '-';
        document.getElementById('metric-residual').textContent = '-';
        document.getElementById('metric-etr').textContent = '-';
    }
}
