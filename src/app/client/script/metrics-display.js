/**
 * Metrics Display Manager - Updates HUD with real-time solver metrics
  */
export class MetricsDisplay {
    constructor(hudElement) {
        this.container = hudElement.querySelector('.metrics-container');
        this.metrics = {};
        
        // Phase 1: Data buffering for the plot
        this.residualHistory = [];
        this.maxHistory = 80; // Number of points displayed on the X-axis
        
        this.createMetricElements();
        
        // Initialize Canvas context
        this.canvas = document.getElementById('convergence-plot');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
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
            
            <div class="metric-row" style="flex-direction: column; align-items: flex-start; gap: 6px; margin: 0px;">
                <span class="metric-label">Convergence Trend (Log Scale):</span>
                <canvas id="convergence-plot" width="240" height="60" 
                    style="width: 100%; height: 70px; background: rgba(0,0,0,0.04); border: 0.5px solid #cccccc; border-radius: 2px;">
                </canvas>
            </div>

            <div class="metric-row">
                <span class="metric-label">Residual:</span>
                <span class="metric-value" id="metric-residual" style="color: var(--hud-warm); font-weight: bold;">-</span>
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
                <span class="metric-label">Nodes / Elements:</span>
                <span class="metric-value"><span id="metric-nodes">-</span> / <span id="metric-elements">-</span></span>
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
        document.getElementById('metric-iteration').textContent = `${iteration} / ${maxIterations}`;
        document.getElementById('metric-residual').textContent = residual.toExponential(3);
        document.getElementById('metric-etr').textContent = this.formatTime(etr);

        // Phase 1: Update Plot Data
        if (residual > 0) {
            this.residualHistory.push(residual);
            if (this.residualHistory.length > this.maxHistory) {
                this.residualHistory.shift();
            }
            this.drawPlot();
        }
    }

    /**
     * Renders the logarithmic sparkline to the HUD canvas
     */
    drawPlot() {
        if (!this.ctx || this.residualHistory.length < 2) return;

        const { ctx, canvas, residualHistory } = this;
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Convert residuals to log10 space for the scale
        const logs = residualHistory.map(v => Math.log10(v));
        const minLog = Math.min(...logs);
        const maxLog = Math.max(...logs);
        const range = maxLog - minLog || 1;

        // Draw background grid (optional "blueprint" style lines)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.lineWidth = 0.5;
        for(let i = 1; i < 4; i++) {
            const gy = (h / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(w, gy);
            ctx.stroke();
        }

        // Draw the convergence line
        ctx.beginPath();
        ctx.setLineDash([]);
        ctx.strokeStyle = "#ff6f55ff";
        ctx.lineWidth = 1;
        ctx.lineJoin = 'round';

        for (let i = 0; i < logs.length; i++) {
            // Horizontal spacing based on history capacity
            const x = (i / (this.maxHistory - 1)) * w;
            // Vertical mapping: lower residuals (lower logs) are plotted higher up or lower down?
            // Convention: Better convergence (lower residual) goes "down" visually.
            const y = h - ((logs[i] - minLog) / range) * (h - 10) - 5;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill area under the curve
        ctx.lineTo((logs.length - 1) / (this.maxHistory - 1) * w, h);
        ctx.lineTo(0, h);
        ctx.fillStyle = "rgba(255, 56, 34, 0.07)";
        ctx.fill();
    }
    
    updateTotalTime(seconds) {
        document.getElementById('metric-total-time').textContent = this.formatTime(seconds);
    }
    
    formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) return '-';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    }
    
    reset() {
        this.updateStatus('Idle');
        this.residualHistory = [];
        document.getElementById('metric-stage').textContent = '-';
        document.getElementById('metric-nodes').textContent = '-';
        document.getElementById('metric-elements').textContent = '-';
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('metric-iteration').textContent = '-';
        document.getElementById('metric-residual').textContent = '-';
        document.getElementById('metric-etr').textContent = '-';
        document.getElementById('metric-total-time').textContent = '-';
        if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}
