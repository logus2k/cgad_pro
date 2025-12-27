/**
 * ConvergencePlot - Real-time line chart showing solver convergence
 * 
 * Displays relative residual vs iteration during solve:
 * - X-axis: Iteration number
 * - Y-axis: Relative residual (log scale)
 * - Live updating during solve
 * - Tooltip shows both residual and relative_residual
 * 
 * Type: LIVE metric (appears on job start, updates during solve)
 * 
 * Location: /src/app/client/script/metrics/widgets/ConvergencePlot.js
 */

import { BaseMetric } from '../BaseMetric.js';

export class ConvergencePlot extends BaseMetric {
    constructor(options = {}) {
        super('convergence-plot', {
            title: 'Convergence',
            defaultWidth: 400,
            defaultHeight: 280,
            position: { top: '120px', right: '400px' },
            ...options
        });
        
        // Data storage
        this.iterations = [];
        this.residuals = [];
        this.relativeResiduals = [];
        this.maxIterations = 0;
        this.solverType = null;
        this.isRunning = false;
    }
    
    /**
     * Bind events - listen for job start and solve progress
     */
    bindEvents() {
        super.bindEvents();
        
        // Listen for job started - show panel and reset
        this._boundHandlers.onJobStarted = (e) => this.onJobStarted(e);
        document.addEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        
        // Listen for solve progress - update chart
        this._boundHandlers.onSolveProgress = (e) => this.onSolveProgress(e);
        document.addEventListener('fem:solveProgress', this._boundHandlers.onSolveProgress);
        
        // Listen for solve complete - mark as finished
        this._boundHandlers.onSolveComplete = (e) => this.onSolveComplete(e);
        document.addEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
    }
    
    /**
     * Handle job started - reset and show panel
     */
    onJobStarted(event) {
        const data = event.detail;
        
        // Reset data
        this.iterations = [];
        this.residuals = [];
        this.relativeResiduals = [];
        this.maxIterations = 0;
        this.solverType = data?.solver_type || null;
        this.isRunning = true;
        this.hasData = true;
        
        // Reset chart to initial state
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getRunningOption(), true);
        }
        
        // Show panel if enabled
        if (this.isEnabled()) {
            this.show();
        }
    }
    
    /**
     * Handle solve progress - add data point and update chart
     */
    onSolveProgress(event) {
        const data = event.detail;
        
        if (!data || data.iteration === undefined) {
            return;
        }
        
        // Update solver type if available
        if (data.solver_type && !this.solverType) {
            this.solverType = data.solver_type;
        }
        
        // Store max iterations
        if (data.max_iterations) {
            this.maxIterations = data.max_iterations;
        }
        
        // Add data point
        this.iterations.push(data.iteration);
        this.residuals.push(data.residual);
        this.relativeResiduals.push(data.relative_residual);
        
        // Update chart
        this.updateChart();
    }
    
    /**
     * Handle solve complete - mark as finished
     */
    onSolveComplete(event) {
        const data = event.detail;
        
        this.isRunning = false;
        
        // Update solver type from complete event if not set
        if (data?.solver_type && !this.solverType) {
            this.solverType = data.solver_type;
        }
        
        // Final chart update with "complete" state
        this.updateChart();
    }
    
    /**
     * Get default chart option (empty/waiting state)
     */
    getDefaultOption() {
        return {
            title: {
                text: 'Awaiting solve...',
                left: 'center',
                top: 'center',
                textStyle: {
                    color: '#999',
                    fontSize: 13,
                    fontWeight: 'normal'
                }
            }
        };
    }
    
    /**
     * Get chart option for running state (before data arrives)
     */
    getRunningOption() {
        return {
            title: {
                text: 'Solving...',
                subtext: this.solverType ? `Solver: ${this.formatSolverType(this.solverType)}` : '',
                left: 'center',
                top: 5,
                textStyle: {
                    fontSize: 13,
                    fontWeight: 'bold',
                    color: '#333'
                },
                subtextStyle: {
                    fontSize: 11,
                    color: '#666'
                }
            },
            grid: {
                left: '12%',
                right: '5%',
                top: 55,
                bottom: 35
            },
            xAxis: {
                type: 'value',
                name: 'Iteration',
                nameLocation: 'center',
                nameGap: 22,
                nameTextStyle: {
                    fontSize: 11,
                    color: '#666'
                },
                axisLabel: {
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
                type: 'log',
                name: 'Rel. Residual',
                nameLocation: 'center',
                nameGap: 40,
                nameTextStyle: {
                    fontSize: 11,
                    color: '#666'
                },
                axisLabel: {
                    fontSize: 10,
                    color: '#666',
                    formatter: (val) => val.toExponential(0)
                },
                splitLine: {
                    lineStyle: {
                        type: 'dashed',
                        color: '#e0e0e0'
                    }
                }
            },
            series: [{
                type: 'line',
                data: [],
                smooth: true,
                symbol: 'none',
                lineStyle: {
                    color: '#5470c6',
                    width: 2
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: 'rgba(84, 112, 198, 0.3)' },
                            { offset: 1, color: 'rgba(84, 112, 198, 0.05)' }
                        ]
                    }
                }
            }]
        };
    }
    
    /**
     * Update chart with current data
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (this.iterations.length === 0) {
            return;
        }
        
        // Build data array for ECharts: [[iter, relResidual, residual], ...]
        const chartData = this.iterations.map((iter, idx) => [
            iter,
            this.relativeResiduals[idx],
            this.residuals[idx]  // Extra dimension for tooltip
        ]);
        
        // Calculate progress
        const currentIter = this.iterations[this.iterations.length - 1];
        const progress = this.maxIterations > 0 
            ? (currentIter / this.maxIterations * 100).toFixed(1)
            : '?';
        
        // Determine title based on state
        let titleText;
        if (this.isRunning) {
            titleText = `Iteration ${currentIter.toLocaleString()}`;
            if (this.maxIterations > 0) {
                titleText += ` / ${this.maxIterations.toLocaleString()} (${progress}%)`;
            }
        } else {
            titleText = `Converged at ${currentIter.toLocaleString()} iterations`;
        }
        
        const option = {
            title: {
                text: titleText,
                subtext: this.solverType ? `Solver: ${this.formatSolverType(this.solverType)}` : '',
                left: 'center',
                top: 5,
                textStyle: {
                    fontSize: 12,
                    fontWeight: 'bold',
                    color: this.isRunning ? '#e6a23c' : '#67c23a'
                },
                subtextStyle: {
                    fontSize: 11,
                    color: '#666'
                }
            },
            tooltip: {
                trigger: 'axis',
                formatter: (params) => {
                    if (!params || !params[0]) return '';
                    const data = params[0].data;
                    const iter = data[0];
                    const relRes = data[1];
                    const res = data[2];
                    return `<strong>Iteration ${iter.toLocaleString()}</strong><br/>
                            Relative: ${relRes.toExponential(3)}<br/>
                            Absolute: ${res.toExponential(3)}`;
                },
                axisPointer: {
                    type: 'cross',
                    lineStyle: {
                        color: '#999',
                        type: 'dashed'
                    }
                }
            },
            grid: {
                left: '12%',
                right: '5%',
                top: 55,
                bottom: 35
            },
            xAxis: {
                type: 'value',
                name: 'Iteration',
                nameLocation: 'center',
                nameGap: 22,
                nameTextStyle: {
                    fontSize: 11,
                    color: '#666'
                },
                min: 0,
                max: this.maxIterations > 0 ? this.maxIterations : undefined,
                axisLabel: {
                    fontSize: 10,
                    color: '#666',
                    formatter: (val) => {
                        if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                        return val;
                    }
                },
                splitLine: {
                    lineStyle: {
                        type: 'dashed',
                        color: '#e0e0e0'
                    }
                }
            },
            yAxis: {
                type: 'log',
                name: 'Rel. Residual',
                nameLocation: 'center',
                nameGap: 40,
                nameTextStyle: {
                    fontSize: 11,
                    color: '#666'
                },
                axisLabel: {
                    fontSize: 10,
                    color: '#666',
                    formatter: (val) => val.toExponential(0)
                },
                splitLine: {
                    lineStyle: {
                        type: 'dashed',
                        color: '#e0e0e0'
                    }
                }
            },
            series: [{
                type: 'line',
                data: chartData,
                smooth: true,
                symbol: 'none',
                sampling: 'lttb',  // Downsample for performance with many points
                lineStyle: {
                    color: this.isRunning ? '#e6a23c' : '#67c23a',
                    width: 2
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: this.isRunning ? [
                            { offset: 0, color: 'rgba(230, 162, 60, 0.3)' },
                            { offset: 1, color: 'rgba(230, 162, 60, 0.05)' }
                        ] : [
                            { offset: 0, color: 'rgba(103, 194, 58, 0.3)' },
                            { offset: 1, color: 'rgba(103, 194, 58, 0.05)' }
                        ]
                    }
                }
            }]
        };
        
        this.chart.setOption(option, false);  // false = merge, not replace
    }
    
    /**
     * Format solver type for display
     */
    formatSolverType(type) {
        const labels = {
            'gpu': 'GPU (CuPy)',
            'numba_cuda': 'Numba CUDA',
            'numba': 'Numba CPU',
            'cpu_multiprocess': 'CPU Multiprocess',
            'cpu_threaded': 'CPU Threaded',
            'cpu': 'CPU (Baseline)'
        };
        return labels[type] || type;
    }
    
    /**
     * Reset to initial state
     */
    reset() {
        this.iterations = [];
        this.residuals = [];
        this.relativeResiduals = [];
        this.maxIterations = 0;
        this.solverType = null;
        this.isRunning = false;
        this.hasData = false;
        
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getDefaultOption(), true);
        }
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        // Remove event listeners
        if (this._boundHandlers.onJobStarted) {
            document.removeEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        }
        if (this._boundHandlers.onSolveProgress) {
            document.removeEventListener('fem:solveProgress', this._boundHandlers.onSolveProgress);
        }
        if (this._boundHandlers.onSolveComplete) {
            document.removeEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        }
        
        super.dispose();
    }
}

export default ConvergencePlot;
