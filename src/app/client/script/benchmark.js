/**
 * Benchmark Panel UI Component
 * 
 * Displays benchmark results in a sortable table with filtering and deletion capabilities.
 * Fetches data from /api/benchmark endpoint.
 * 
 * Location: /src/app/client/script/benchmark.js
 */

export class BenchmarkPanel {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId) || document.querySelector(containerId);
        if (!this.container) {
            console.error(`[Benchmark] Container not found: ${containerId}`);
            return;
        }
        
        this.options = {
            apiBase: options.apiBase || '',
            pollInterval: options.pollInterval || 30000, // 30 seconds
            autoRefresh: options.autoRefresh !== false,
            ...options
        };
        
        this.records = [];
        this.selectedRecords = new Set();
        this.sortColumn = 'timestamp';
        this.sortOrder = 'desc';
        this.filterSolver = '';
        this.filterModel = '';
        this.serverConfig = null;
        
        this.pollTimer = null;
        
        this.init();
    }
    
    async init() {
        this.render();
        await this.fetchData();
        
        if (this.options.autoRefresh) {
            this.startPolling();
        }
    }
    
    render() {
        this.container.innerHTML = `
            <div class="benchmark-controls">
                <div class="benchmark-controls-left">
                    <select class="benchmark-filter" id="benchmark-filter-solver">
                        <option value="">All Solvers</option>
                    </select>
                    <select class="benchmark-filter" id="benchmark-filter-model">
                        <option value="">All Models</option>
                    </select>
                </div>
                <div class="benchmark-controls-right">
                    <span class="benchmark-server-info" id="benchmark-server-info"></span>
                    <button class="benchmark-btn benchmark-btn-secondary" id="benchmark-refresh-btn">
                        Refresh
                    </button>
                    <button class="benchmark-btn benchmark-btn-danger" id="benchmark-delete-btn" disabled>
                        Delete Selected
                    </button>
                </div>
            </div>
            
            <div class="benchmark-summary" id="benchmark-summary">
                <div class="benchmark-stat">
                    <span class="benchmark-stat-value" id="stat-total">0</span>
                    <span class="benchmark-stat-label">Records</span>
                </div>
                <div class="benchmark-stat">
                    <span class="benchmark-stat-value" id="stat-solvers">0</span>
                    <span class="benchmark-stat-label">Solvers</span>
                </div>
                <div class="benchmark-stat">
                    <span class="benchmark-stat-value" id="stat-models">0</span>
                    <span class="benchmark-stat-label">Models</span>
                </div>
                <div class="benchmark-stat">
                    <span class="benchmark-stat-value" id="stat-best-time">-</span>
                    <span class="benchmark-stat-label">Best Time</span>
                </div>
            </div>
            
            <div class="benchmark-table-container" id="benchmark-table-container">
                <div class="benchmark-loading">
                    <div class="benchmark-loading-spinner"></div>
                    <div>Loading benchmark data...</div>
                </div>
            </div>
        `;
        
        this.bindEvents();
    }
    
    bindEvents() {
        // Refresh button
        const refreshBtn = this.container.querySelector('#benchmark-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.fetchData());
        }
        
        // Delete button
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => this.deleteSelected());
        }
        
        // Solver filter
        const solverFilter = this.container.querySelector('#benchmark-filter-solver');
        if (solverFilter) {
            solverFilter.addEventListener('change', (e) => {
                this.filterSolver = e.target.value;
                this.renderTable();
            });
        }
        
        // Model filter
        const modelFilter = this.container.querySelector('#benchmark-filter-model');
        if (modelFilter) {
            modelFilter.addEventListener('change', (e) => {
                this.filterModel = e.target.value;
                this.renderTable();
            });
        }
    }
    
    async fetchData() {
        try {
            const response = await fetch(`${this.options.apiBase}/api/benchmark?sort_by=${this.sortColumn}&sort_order=${this.sortOrder}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.records = data.records || [];
            
            // Fetch summary
            const summaryResponse = await fetch(`${this.options.apiBase}/api/benchmark/summary`);
            if (summaryResponse.ok) {
                const summary = await summaryResponse.json();
                this.serverConfig = summary.server_config;
                this.updateSummary(summary);
                this.updateFilters(summary);
            }
            
            this.renderTable();
            
        } catch (error) {
            console.error('[Benchmark] Failed to fetch data:', error);
            this.renderError(error.message);
        }
    }
    
    updateSummary(summary) {
        const statTotal = this.container.querySelector('#stat-total');
        const statSolvers = this.container.querySelector('#stat-solvers');
        const statModels = this.container.querySelector('#stat-models');
        const statBestTime = this.container.querySelector('#stat-best-time');
        const serverInfo = this.container.querySelector('#benchmark-server-info');
        
        if (statTotal) statTotal.textContent = summary.total_records || 0;
        if (statSolvers) statSolvers.textContent = (summary.solver_types || []).length;
        if (statModels) statModels.textContent = (summary.models || []).length;
        
        // Find best time across all solvers
        if (statBestTime && summary.best_times) {
            const times = Object.values(summary.best_times).map(b => b.time).filter(t => t);
            if (times.length > 0) {
                const best = Math.min(...times);
                statBestTime.textContent = this.formatTime(best);
            } else {
                statBestTime.textContent = '-';
            }
        }
        
        // Server info
        if (serverInfo && summary.server_config) {
            const cfg = summary.server_config;
            const gpu = cfg.gpu_model ? ` | ${cfg.gpu_model}` : '';
            serverInfo.textContent = `${cfg.cpu_model} (${cfg.cpu_cores} cores)${gpu}`;
            serverInfo.title = `Server: ${cfg.hostname}\nOS: ${cfg.os}\nRAM: ${cfg.ram_gb} GB`;
        }
    }
    
    updateFilters(summary) {
        const solverFilter = this.container.querySelector('#benchmark-filter-solver');
        const modelFilter = this.container.querySelector('#benchmark-filter-model');
        
        if (solverFilter && summary.solver_types) {
            const currentValue = solverFilter.value;
            solverFilter.innerHTML = '<option value="">All Solvers</option>';
            summary.solver_types.sort().forEach(solver => {
                const option = document.createElement('option');
                option.value = solver;
                option.textContent = this.formatSolverName(solver);
                solverFilter.appendChild(option);
            });
            solverFilter.value = currentValue;
        }
        
        if (modelFilter && summary.models) {
            const currentValue = modelFilter.value;
            modelFilter.innerHTML = '<option value="">All Models</option>';
            summary.models.sort().forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelFilter.appendChild(option);
            });
            modelFilter.value = currentValue;
        }
    }
    
    renderTable() {
        const container = this.container.querySelector('#benchmark-table-container');
        if (!container) return;
        
        // Filter records
        let filtered = this.records;
        if (this.filterSolver) {
            filtered = filtered.filter(r => r.solver_type === this.filterSolver);
        }
        if (this.filterModel) {
            filtered = filtered.filter(r => r.model_name === this.filterModel);
        }
        
        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="benchmark-empty">
                    <div class="benchmark-empty-icon">📊</div>
                    <div class="benchmark-empty-text">
                        ${this.records.length === 0 
                            ? 'No benchmark records yet. Run a solver to record results.' 
                            : 'No records match the current filters.'}
                    </div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th data-column="select" style="width: 30px;"></th>
                        <th data-column="model_name" class="${this.getSortClass('model_name')}">Model</th>
                        <th data-column="solver_type" class="${this.getSortClass('solver_type')}">Solver</th>
                        <th data-column="model_nodes" class="${this.getSortClass('model_nodes')}">Nodes</th>
                        <th data-column="model_elements" class="${this.getSortClass('model_elements')}">Elements</th>
                        <th data-column="total_time" class="${this.getSortClass('total_time')}">Total Time</th>
                        <th data-column="timings.assemble_system" class="${this.getSortClass('timings.assemble_system')}">Assembly</th>
                        <th data-column="timings.solve_system" class="${this.getSortClass('timings.solve_system')}">Solve</th>
                        <th data-column="iterations" class="${this.getSortClass('iterations')}">Iterations</th>
                        <th data-column="peak_ram" class="${this.getSortClass('peak_ram')}">Peak RAM</th>
                        <th data-column="peak_vram" class="${this.getSortClass('peak_vram')}">Peak VRAM</th>
                        <th data-column="converged" class="${this.getSortClass('converged')}">Status</th>
                        <th data-column="timestamp" class="${this.getSortClass('timestamp')}">Date</th>
                    </tr>
                </thead>
                <tbody>
                    ${filtered.map(record => this.renderRow(record)).join('')}
                </tbody>
            </table>
        `;
        
        // Bind table events
        this.bindTableEvents();
    }
    
    renderRow(record) {
        const totalTime = record.timings?.total_program_time || 0;
        const timeClass = this.getTimeClass(totalTime);
        const isSelected = this.selectedRecords.has(record.id);
        
        // Extract memory data (with fallback for old records)
        const peakRam = record.memory?.peak_ram_mb;
        const peakVram = record.memory?.peak_vram_mb;
        
        return `
            <tr class="${isSelected ? 'selected' : ''}" data-id="${record.id}">
                <td>
                    <input type="checkbox" class="benchmark-checkbox" 
                           data-id="${record.id}" 
                           ${isSelected ? 'checked' : ''}>
                </td>
                <td class="benchmark-cell-model" title="${record.model_name}">${record.model_name}</td>
                <td class="benchmark-cell-solver">
                    <span class="solver-badge ${this.getSolverBadgeClass(record.solver_type)}">
                        ${this.formatSolverName(record.solver_type)}
                    </span>
                </td>
                <td>${this.formatNumber(record.model_nodes)}</td>
                <td>${this.formatNumber(record.model_elements)}</td>
                <td class="benchmark-cell-time ${timeClass}">${this.formatTime(totalTime)}</td>
                <td class="benchmark-cell-time">${this.formatTime(record.timings?.assemble_system)}</td>
                <td class="benchmark-cell-time">${this.formatTime(record.timings?.solve_system)}</td>
                <td>${this.formatNumber(record.iterations)}</td>
                <td class="benchmark-cell-memory">${this.formatMemory(peakRam)}</td>
                <td class="benchmark-cell-memory ${this.getVramClass(peakVram)}">${this.formatMemory(peakVram)}</td>
                <td class="benchmark-cell-status">
                    <span class="${record.converged ? 'converged' : 'failed'}">
                        ${record.converged ? '✓' : '✗'}
                    </span>
                </td>
                <td class="benchmark-cell-timestamp">${this.formatTimestamp(record.timestamp)}</td>
            </tr>
        `;
    }
    
    bindTableEvents() {
        const table = this.container.querySelector('.benchmark-table');
        if (!table) return;
        
        // Header click for sorting
        table.querySelectorAll('th[data-column]').forEach(th => {
            if (th.dataset.column === 'select') return;
            
            th.addEventListener('click', () => {
                const column = th.dataset.column;
                if (this.sortColumn === column) {
                    this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
                } else {
                    this.sortColumn = column;
                    this.sortOrder = column === 'timestamp' ? 'desc' : 'asc';
                }
                this.fetchData();
            });
        });
        
        // Checkbox click for selection
        table.querySelectorAll('.benchmark-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const id = e.target.dataset.id;
                if (e.target.checked) {
                    this.selectedRecords.add(id);
                } else {
                    this.selectedRecords.delete(id);
                }
                this.updateDeleteButton();
                
                // Update row highlight
                const row = e.target.closest('tr');
                if (row) {
                    row.classList.toggle('selected', e.target.checked);
                }
            });
        });
    }
    
    updateDeleteButton() {
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.disabled = this.selectedRecords.size === 0;
            deleteBtn.textContent = this.selectedRecords.size > 0 
                ? `Delete Selected (${this.selectedRecords.size})`
                : 'Delete Selected';
        }
    }
    
    async deleteSelected() {
        if (this.selectedRecords.size === 0) return;
        
        const count = this.selectedRecords.size;
        if (!confirm(`Delete ${count} selected record(s)?`)) return;
        
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.disabled = true;
            deleteBtn.textContent = 'Deleting...';
        }
        
        try {
            const deletePromises = Array.from(this.selectedRecords).map(id =>
                fetch(`${this.options.apiBase}/api/benchmark/${id}`, { method: 'DELETE' })
            );
            
            await Promise.all(deletePromises);
            
            this.selectedRecords.clear();
            await this.fetchData();
            
            console.log(`[Benchmark] Deleted ${count} record(s)`);
            
        } catch (error) {
            console.error('[Benchmark] Delete failed:', error);
            alert('Failed to delete some records. Please try again.');
        }
        
        this.updateDeleteButton();
    }
    
    renderError(message) {
        const container = this.container.querySelector('#benchmark-table-container');
        if (container) {
            container.innerHTML = `
                <div class="benchmark-empty">
                    <div class="benchmark-empty-icon">⚠️</div>
                    <div class="benchmark-empty-text">
                        Failed to load benchmark data: ${message}
                    </div>
                    <button class="benchmark-btn benchmark-btn-secondary" 
                            style="margin-top: 10px;"
                            onclick="this.closest('.benchmark-container').querySelector('#benchmark-refresh-btn').click()">
                        Retry
                    </button>
                </div>
            `;
        }
    }
    
    // Helper methods
    
    getSortClass(column) {
        if (this.sortColumn !== column) return '';
        return this.sortOrder === 'asc' ? 'sorted-asc' : 'sorted-desc';
    }
    
    getSolverBadgeClass(solver) {
        const classes = {
            'gpu': 'gpu',
            'numba_cuda': 'numba-cuda',
            'numba': 'numba',
            'cpu_multiprocess': 'cpu-multiprocess',
            'cpu_threaded': 'cpu-threaded',
            'cpu': 'cpu'
        };
        return classes[solver] || 'cpu';
    }
    
    formatSolverName(solver) {
        const names = {
            'gpu': 'GPU',
            'numba_cuda': 'Numba CUDA',
            'numba': 'Numba',
            'cpu_multiprocess': 'Multiprocess',
            'cpu_threaded': 'Threaded',
            'cpu': 'CPU'
        };
        return names[solver] || solver;
    }
    
    formatTime(seconds) {
        if (seconds === undefined || seconds === null) return '-';
        if (seconds < 0.01) return '<0.01s';
        if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
        if (seconds < 60) return `${seconds.toFixed(2)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${mins}m ${secs}s`;
    }
    
    formatNumber(num) {
        if (num === undefined || num === null) return '-';
        return num.toLocaleString();
    }
    
    formatTimestamp(timestamp) {
        if (!timestamp) return '-';
        const date = new Date(timestamp);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    /**
     * Format memory value in MB to human-readable string
     * @param {number} mb - Memory in megabytes
     * @returns {string} Formatted string (e.g., "512 MB", "2.5 GB", "-")
     */
    formatMemory(mb) {
        if (mb === undefined || mb === null || mb === 0) return '-';
        if (mb < 1024) return `${Math.round(mb)} MB`;
        return `${(mb / 1024).toFixed(2)} GB`;
    }
    
    /**
     * Get CSS class for VRAM cell based on usage
     * @param {number} mb - VRAM in megabytes
     * @returns {string} CSS class name
     */
    getVramClass(mb) {
        if (mb === undefined || mb === null || mb === 0) return '';
        if (mb > 16000) return 'memory-high';    // > 16 GB
        if (mb > 8000) return 'memory-medium';   // > 8 GB
        return 'memory-low';                      // <= 8 GB
    }
    
    getTimeClass(seconds) {
        if (seconds < 10) return 'fast';
        if (seconds < 60) return 'medium';
        return 'slow';
    }
    
    // Polling
    
    startPolling() {
        this.stopPolling();
        this.pollTimer = setInterval(() => this.fetchData(), this.options.pollInterval);
    }
    
    stopPolling() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
    }
    
    // Public methods
    
    refresh() {
        return this.fetchData();
    }
    
    destroy() {
        this.stopPolling();
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// Auto-initialize if container exists
export function initBenchmarkPanel(containerId = '.benchmark-container', options = {}) {
    const panel = new BenchmarkPanel(containerId, options);
    return panel;
}

export default BenchmarkPanel;
