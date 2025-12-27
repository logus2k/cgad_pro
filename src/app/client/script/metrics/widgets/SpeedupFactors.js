/**
 * SpeedupFactors - Horizontal bar chart comparing solver performance
 * 
 * Displays speedup factors for all solver implementations:
 * - Shows all 6 solver types (never hides rows)
 * - Solvers with data: colored bar + time
 * - Solvers without data: "N/A" with dimmed bar
 * - When CPU baseline exists: shows speedup factors (e.g., "77.2x")
 * - Footer shows best performer's hardware info
 * 
 * Data source: Benchmark API (/api/benchmark/results)
 * 
 * Type: POST-SOLVE metric (updates when solve completes)
 * 
 * Location: /src/app/client/script/metrics/widgets/SpeedupFactors.js
 */

import { BaseMetric } from '../BaseMetric.js';

// All solver types in display order (fastest typically at top)
const SOLVER_ORDER = [
    'gpu',
    'numba_cuda', 
    'numba',
    'cpu_multiprocess',
    'cpu_threaded',
    'cpu'
];

// Solver display configuration
const SOLVER_CONFIG = {
    gpu: {
        label: 'GPU (CuPy)',
        color: '#ee6666',
        isBaseline: false
    },
    numba_cuda: {
        label: 'Numba CUDA',
        color: '#fc8452',
        isBaseline: false
    },
    numba: {
        label: 'Numba CPU',
        color: '#fac858',
        isBaseline: false
    },
    cpu_multiprocess: {
        label: 'CPU Multiprocess',
        color: '#91cc75',
        isBaseline: false
    },
    cpu_threaded: {
        label: 'CPU Threaded',
        color: '#73c0de',
        isBaseline: false
    },
    cpu: {
        label: 'CPU (Baseline)',
        color: '#5470c6',
        isBaseline: true
    }
};

export class SpeedupFactors extends BaseMetric {
    constructor(options = {}) {
        super('speedup-factors', {
            title: 'Speedup Factors',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '420px', right: '20px' },
            ...options
        });
        
        // Data storage
        this.currentMesh = null;
        this.benchmarkData = null;
        this.solverResults = {};  // { solver_type: { time, record } }
        this.bestSolver = null;
        this.baselineTime = null;
    }
    
    /**
     * Bind events
     */
    bindEvents() {
        super.bindEvents();
        
        // Listen for job started - get mesh name
        this._boundHandlers.onJobStarted = (e) => this.onJobStarted(e);
        document.addEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        
        // Listen for solve complete - refresh data
        this._boundHandlers.onSolveComplete = (e) => this.onSolveComplete(e);
        document.addEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        
        // Listen for mesh selected - update mesh context
        this._boundHandlers.onMeshSelected = (e) => this.onMeshSelected(e);
        document.addEventListener('meshSelected', this._boundHandlers.onMeshSelected);
    }
    
    /**
     * Handle mesh selected from gallery
     */
    onMeshSelected(event) {
        const mesh = event.detail;
        if (mesh && mesh.file) {
            // Extract filename from path
            const filename = mesh.file.split('/').pop();
            this.currentMesh = {
                file: filename,
                name: mesh.name || filename,
                nodes: mesh.nodes,
                elements: mesh.elements
            };
            
            // Fetch benchmark data for this mesh
            this.fetchBenchmarkData();
        }
    }
    
    /**
     * Handle job started
     */
    onJobStarted(event) {
        const data = event.detail;
        
        // Try to extract mesh info
        if (data && data.mesh_file) {
            const filename = data.mesh_file.split('/').pop();
            this.currentMesh = {
                file: filename,
                name: filename,
                nodes: null,
                elements: null
            };
        }
    }
    
    /**
     * Handle solve complete - refresh benchmark data
     */
    onSolveComplete(event) {
        const data = event.detail;
        
        // Update mesh info if available
        if (data && data.mesh_info) {
            if (this.currentMesh) {
                this.currentMesh.nodes = data.mesh_info.nodes;
                this.currentMesh.elements = data.mesh_info.elements;
            }
        }
        
        // Refresh benchmark data (new result may have been added)
        this.fetchBenchmarkData();
        
        // Show panel if enabled
        if (this.isEnabled()) {
            this.show();
        }
    }
    
    /**
     * Fetch benchmark data from API
     */
    async fetchBenchmarkData() {
        try {
            // Determine API base based on environment
            const apiBase = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
                ? '' 
                : '/fem';
            
            const response = await fetch(`${apiBase}/api/benchmark`);
            if (!response.ok) {
                console.warn('[SpeedupFactors] Failed to fetch benchmark data:', response.status);
                return;
            }
            
            const data = await response.json();
            // API returns { count, records }
            this.benchmarkData = { records: data.records || [] };
            
            // Process data for current mesh
            this.processData();
            
        } catch (error) {
            console.error('[SpeedupFactors] Error fetching benchmark data:', error);
        }
    }
    
    /**
     * Process benchmark data for current mesh
     */
    processData() {
        // Reset results
        this.solverResults = {};
        this.bestSolver = null;
        this.baselineTime = null;
        
        if (!this.benchmarkData || !this.benchmarkData.records) {
            this.updateChart();
            return;
        }
        
        const records = this.benchmarkData.records;
        
        // Filter records for current mesh (if known)
        let relevantRecords = records;
        if (this.currentMesh && this.currentMesh.file) {
            relevantRecords = records.filter(r => 
                r.model_file === this.currentMesh.file || 
                r.model_name === this.currentMesh.file
            );
        }
        
        // If no records for current mesh, use all records (show general benchmark)
        if (relevantRecords.length === 0) {
            relevantRecords = records;
        }
        
        // Group by solver_type, keep fastest run for each
        for (const record of relevantRecords) {
            const solverType = record.solver_type;
            const time = record.timings?.total_program_time;
            
            if (!solverType || time === undefined) continue;
            
            if (!this.solverResults[solverType] || time < this.solverResults[solverType].time) {
                this.solverResults[solverType] = {
                    time: time,
                    record: record
                };
            }
        }
        
        // Find baseline time (CPU)
        if (this.solverResults.cpu) {
            this.baselineTime = this.solverResults.cpu.time;
        }
        
        // Find best (fastest) solver
        let bestTime = Infinity;
        for (const [solver, data] of Object.entries(this.solverResults)) {
            if (data.time < bestTime) {
                bestTime = data.time;
                this.bestSolver = solver;
            }
        }
        
        // Update mesh info from records if not set
        if (this.currentMesh && !this.currentMesh.nodes && relevantRecords.length > 0) {
            const firstRecord = relevantRecords[0];
            this.currentMesh.nodes = firstRecord.model_nodes;
            this.currentMesh.elements = firstRecord.model_elements;
            this.currentMesh.name = firstRecord.model_name;
        }
        
        this.hasData = Object.keys(this.solverResults).length > 0;
        this.updateChart();
    }
    
    /**
     * Get default chart option (empty state)
     */
    getDefaultOption() {
        return {
            title: {
                text: 'No benchmark data',
                subtext: 'Run solves to collect data',
                left: 'center',
                top: 'center',
                textStyle: {
                    color: '#999',
                    fontSize: 13,
                    fontWeight: 'normal'
                },
                subtextStyle: {
                    color: '#bbb',
                    fontSize: 11
                }
            }
        };
    }
    
    /**
     * Update chart with current data
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        // Build data arrays (in reverse order for ECharts - bottom to top)
        const categories = [];
        const values = [];
        const itemStyles = [];
        const labels = [];
        
        // Process solvers in reverse order (so first appears at top)
        for (let i = SOLVER_ORDER.length - 1; i >= 0; i--) {
            const solverType = SOLVER_ORDER[i];
            const config = SOLVER_CONFIG[solverType];
            const result = this.solverResults[solverType];
            
            categories.push(config.label);
            
            if (result) {
                const time = result.time;
                values.push(time);
                
                // Style for available data (no border highlight)
                const isBest = solverType === this.bestSolver;
                itemStyles.push({
                    color: config.color
                });
                
                // Build label with time and speedup
                let labelText = this.formatTime(time);
                if (this.baselineTime && !config.isBaseline) {
                    const speedup = this.baselineTime / time;
                    labelText += ` (${speedup.toFixed(1)}x)`;
                }
                if (isBest) {
                    labelText += ' ⭐';
                }
                labels.push(labelText);
                
            } else {
                // No data - show as N/A
                values.push(0.001);  // Small value for visual placeholder
                itemStyles.push({
                    color: '#e0e0e0',
                    decal: {
                        symbol: 'rect',
                        symbolSize: 1,
                        color: '#ccc',
                        dashArrayX: [1, 0],
                        dashArrayY: [4, 3],
                        rotation: 0.5
                    }
                });
                labels.push('N/A');
            }
        }
        
        // Find max time for axis scaling
        const maxTime = Math.max(...Object.values(this.solverResults).map(r => r.time), 0.1);
        
        // Build title - show best solver info
        let titleText = 'Solver Comparison';
        let subtitleText = '';
        
        if (this.bestSolver && this.solverResults[this.bestSolver]) {
            const bestConfig = SOLVER_CONFIG[this.bestSolver];
            const bestTime = this.solverResults[this.bestSolver].time;
            titleText = `${bestConfig.label} - ${this.formatTime(bestTime)} ⭐`;
            
            // Build subtitle with hardware + mesh info
            const subtitleParts = [];
            
            // Hardware info
            const bestRecord = this.solverResults[this.bestSolver].record;
            if (bestRecord?.server_config) {
                const gpuShort = this.shortenGpuName(bestRecord.server_config.gpu_model);
                const cpuShort = this.shortenCpuName(bestRecord.server_config.cpu_model);
                subtitleParts.push(`${gpuShort} + ${cpuShort}`);
            }
            
            // Mesh info
            if (this.currentMesh) {
                let meshInfo = this.currentMesh.name || this.currentMesh.file;
                if (this.currentMesh.nodes) {
                    meshInfo += ` (${this.currentMesh.nodes.toLocaleString()} nodes)`;
                }
                subtitleParts.push(meshInfo);
            }
            
            subtitleText = subtitleParts.join(' | ');
        } else if (this.currentMesh) {
            // No best solver yet, just show mesh info
            titleText = 'Solver Comparison';
            subtitleText = this.currentMesh.name || this.currentMesh.file;
            if (this.currentMesh.nodes) {
                subtitleText += ` (${this.currentMesh.nodes.toLocaleString()} nodes)`;
            }
        }
        
        const option = {
            title: {
                text: titleText,
                subtext: subtitleText,
                left: 'center',
                top: 5,
                textStyle: {
                    fontSize: 13,
                    fontWeight: 'bold',
                    color: '#333'
                },
                subtextStyle: {
                    fontSize: 10,
                    color: '#666'
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                formatter: (params) => {
                    const param = params[0];
                    const idx = categories.length - 1 - param.dataIndex;
                    const solverType = SOLVER_ORDER[idx];
                    const result = this.solverResults[solverType];
                    
                    if (!result) {
                        return `<strong>${param.name}</strong><br/>No data available`;
                    }
                    
                    let html = `<strong>${param.name}</strong><br/>`;
                    html += `Time: ${this.formatTime(result.time)}<br/>`;
                    
                    if (this.baselineTime && solverType !== 'cpu') {
                        const speedup = this.baselineTime / result.time;
                        html += `Speedup: ${speedup.toFixed(2)}x vs CPU<br/>`;
                    }
                    
                    if (result.record) {
                        html += `Iterations: ${result.record.iterations?.toLocaleString() || 'N/A'}`;
                    }
                    
                    return html;
                }
            },
            grid: {
                left: '3%',
                right: '22%',
                top: 50,
                bottom: 10,
                containLabel: true
            },
            xAxis: {
                type: 'value',
                max: maxTime * 1.1,
                axisLabel: {
                    formatter: (val) => this.formatTimeShort(val),
                    fontSize: 10,
                    color: '#666'
                },
                splitLine: {
                    lineStyle: {
                        type: 'dashed',
                        color: '#e0e0e0'
                    }
                }
            },
            yAxis: {
                type: 'category',
                data: categories,
                axisLabel: {
                    fontSize: 11,
                    color: '#333'
                },
                axisLine: { show: false },
                axisTick: { show: false }
            },
            series: [{
                name: 'Time',
                type: 'bar',
                data: values.map((val, idx) => ({
                    value: val,
                    itemStyle: itemStyles[idx],
                    label: {
                        show: true,
                        position: 'right',
                        formatter: labels[idx],
                        fontSize: 10,
                        color: labels[idx] === 'N/A' ? '#999' : '#333'
                    }
                })),
                barWidth: '55%'
            }]
        };
        
        this.chart.setOption(option, true);
    }
    
    /**
     * Shorten GPU name for display
     */
    shortenGpuName(name) {
        if (!name) return 'Unknown GPU';
        
        // "NVIDIA GeForce RTX 4090" -> "RTX 4090"
        return name
            .replace('NVIDIA GeForce ', '')
            .replace('NVIDIA ', '')
            .replace('AMD Radeon ', '')
            .trim();
    }
    
    /**
     * Shorten CPU name for display
     */
    shortenCpuName(name) {
        if (!name) return 'Unknown CPU';
        
        // "13th Gen Intel(R) Core(TM) i9-13900K" -> "i9-13900K"
        const match = name.match(/i[3579]-\d{4,5}[A-Z]*/i);
        if (match) return match[0];
        
        // "AMD Ryzen 9 5950X" -> "Ryzen 9 5950X"
        if (name.includes('Ryzen')) {
            return name.replace('AMD ', '').trim();
        }
        
        // Fallback: truncate
        return name.length > 20 ? name.substring(0, 20) + '...' : name;
    }
    
    /**
     * Format time value
     */
    formatTime(seconds) {
        if (seconds === undefined || seconds === null || isNaN(seconds)) {
            return 'N/A';
        }
        
        if (seconds < 0.001) {
            return `${(seconds * 1000000).toFixed(0)}µs`;
        } else if (seconds < 1) {
            return `${(seconds * 1000).toFixed(1)}ms`;
        } else if (seconds < 60) {
            return `${seconds.toFixed(2)}s`;
        } else {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(1);
            return `${mins}m ${secs}s`;
        }
    }
    
    /**
     * Format time for axis labels (shorter)
     */
    formatTimeShort(seconds) {
        if (seconds < 1) {
            return `${(seconds * 1000).toFixed(0)}ms`;
        } else if (seconds < 60) {
            return `${seconds.toFixed(1)}s`;
        } else {
            return `${(seconds / 60).toFixed(1)}m`;
        }
    }
    
    /**
     * Reset to initial state
     */
    reset() {
        this.currentMesh = null;
        this.benchmarkData = null;
        this.solverResults = {};
        this.bestSolver = null;
        this.baselineTime = null;
        this.hasData = false;
        
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getDefaultOption(), true);
        }
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this._boundHandlers.onJobStarted) {
            document.removeEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        }
        if (this._boundHandlers.onSolveComplete) {
            document.removeEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        }
        if (this._boundHandlers.onMeshSelected) {
            document.removeEventListener('meshSelected', this._boundHandlers.onMeshSelected);
        }
        
        super.dispose();
    }
}

export default SpeedupFactors;
