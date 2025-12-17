// ui_controls.js - Add to your Settings panel

class FEMControlPanel {
  constructor(issSystem) {
    this.issSystem = issSystem;
    this.createPanel();
  }
  
  createPanel() {
    const panel = document.createElement('div');
    panel.id = 'fem-controls';
    panel.innerHTML = `
      <div class="panel-section">
        <h3>FEM Thermal Analysis</h3>
        
        <div class="solver-select">
          <label>Solver:</label>
          <select id="solver-type">
            <option value="cpu">CPU (NumPy)</option>
            <option value="gpu" selected>GPU (CuPy)</option>
            <option value="numba">Numba JIT</option>
            <option value="threaded">Free Threading</option>
          </select>
        </div>
        
        <div class="mesh-density">
          <label>Mesh Density:</label>
          <select id="mesh-density">
            <option value="5000">5k elements (Fast)</option>
            <option value="20000" selected>20k elements</option>
            <option value="50000">50k elements (Detailed)</option>
          </select>
        </div>
        
        <button id="run-fem" class="btn-primary">Run Analysis</button>
        <button id="stop-fem" class="btn-secondary" disabled>Stop</button>
        
        <div class="progress-container">
          <div id="fem-progress" class="progress-bar"></div>
          <span id="progress-text">0%</span>
        </div>
        
        <div class="results-display">
          <h4>Performance Metrics</h4>
          <table id="results-table">
            <tr><th>Solver</th><th>Time (s)</th><th>Speedup</th></tr>
          </table>
          
          <canvas id="speedup-chart" width="300" height="150"></canvas>
        </div>
        
        <div class="temp-legend">
          <h4>Temperature Scale</h4>
          <div class="gradient-bar"></div>
          <div class="legend-labels">
            <span>200K</span>
            <span>300K</span>
            <span>400K</span>
          </div>
        </div>
      </div>
    `;
    
    document.querySelector('.settings-panel').appendChild(panel);
    this.attachEventListeners();
  }
  
  attachEventListeners() {
    document.getElementById('run-fem').addEventListener('click', () => {
      const solver = document.getElementById('solver-type').value;
      const density = document.getElementById('mesh-density').value;
      
      this.issSystem.socket.emit('run_solver', {
        solver: solver,
        elements: parseInt(density)
      });
      
      document.getElementById('run-fem').disabled = true;
      document.getElementById('stop-fem').disabled = false;
    });
  }
  
  updateProgressBar(progress) {
    const bar = document.getElementById('fem-progress');
    const text = document.getElementById('progress-text');
    
    bar.style.width = `${progress * 100}%`;
    text.textContent = `${(progress * 100).toFixed(1)}%`;
  }
}
