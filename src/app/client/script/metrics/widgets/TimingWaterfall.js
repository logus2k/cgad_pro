/**
 * TimingWaterfall - Horizontal bar chart showing solver stage timings
 * 
 * Displays time spent in each FEM solver stage:
 * - load_mesh
 * - assemble_system
 * - apply_bc
 * - solve_system
 * - compute_derived (post-process)
 * 
 * Location: /src/app/client/script/metrics/widgets/TimingWaterfall.js
 */

import { BaseMetric } from '../BaseMetric.js';

// Stage display configuration
const STAGE_CONFIG = {
    load_mesh: {
        label: 'Load Mesh',
        color: '#5470c6'
    },
    assemble_system: {
        label: 'Assemble',
        color: '#91cc75'
    },
    apply_bc: {
        label: 'Apply BC',
        color: '#fac858'
    },
    solve_system: {
        label: 'Solve',
        color: '#ee6666'
    },
    compute_derived: {
        label: 'Post-Process',
        color: '#73c0de'
    }
};

// Order of stages in the chart (top to bottom)
const STAGE_ORDER = [
    'load_mesh',
    'assemble_system',
    'apply_bc',
    'solve_system',
    'compute_derived'
];

export class TimingWaterfall extends BaseMetric {
    constructor(options = {}) {
        super('timing-waterfall', {
            title: 'Timing Breakdown',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '120px', right: '20px' },
            ...options
        });
        
        this.timingData = null;
        this.solverType = null;
    }
    
    /**
     * Bind events - listen for solve_complete
     */
    bindEvents() {
        super.bindEvents();
        
        // Listen for solve complete event
        this._boundHandlers.onSolveComplete = (e) => this.onSolveComplete(e);
        document.addEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        
        // Listen for solve start to reset
        this._boundHandlers.onSolveStart = (e) => this.onSolveStart(e);
        document.addEventListener('fem:jobStarted', this._boundHandlers.onSolveStart);
    }
    
    /**
     * Handle solve start - reset the chart
     */
    onSolveStart(event) {
        this.reset();
    }
    
    /**
     * Handle solve complete - update with timing data
     */
    onSolveComplete(event) {
        const data = event.detail;
        
        if (!data || !data.timing_metrics) {
            console.warn('[TimingWaterfall] No timing_metrics in solve_complete data');
            return;
        }
        
        this.solverType = data.solver_type || 'unknown';
        this.update(data.timing_metrics);
        
        // Show panel if metric is enabled
        if (this.isEnabled()) {
            this.show();
        }
    }
    
    /**
     * Get default chart option (empty state)
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
     * Update chart with timing data
     * @param {object} timingMetrics - Timing metrics from solve_complete
     */
    update(timingMetrics) {
        this.timingData = timingMetrics;
        this.hasData = true;
        
        if (!this.chart || this.chart.isDisposed()) {
            console.warn('[TimingWaterfall] Chart not available');
            return;
        }
        
        const totalTime = timingMetrics.total_program_time || 0;
        
        // Build data arrays
        const categories = [];
        const values = [];
        const colors = [];
        const percentages = [];
        
        // Process stages in reverse order (so first stage appears at top)
        for (let i = STAGE_ORDER.length - 1; i >= 0; i--) {
            const stageKey = STAGE_ORDER[i];
            const config = STAGE_CONFIG[stageKey];
            const time = timingMetrics[stageKey] || 0;
            const percent = totalTime > 0 ? (time / totalTime * 100) : 0;
            
            categories.push(config.label);
            values.push(time);
            colors.push(config.color);
            percentages.push(percent);
        }
        
        // Find max value for axis scaling
        const maxValue = Math.max(...values) * 1.15; // 15% padding
        
        const option = {
            title: {
                text: `Total: ${this.formatTime(totalTime)}`,
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
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                },
                formatter: (params) => {
                    const param = params[0];
                    const idx = categories.length - 1 - param.dataIndex;
                    const stageKey = STAGE_ORDER[idx];
                    const time = values[categories.length - 1 - param.dataIndex];
                    const percent = percentages[categories.length - 1 - param.dataIndex];
                    return `<strong>${param.name}</strong><br/>
                            Time: ${this.formatTime(time)}<br/>
                            Percent: ${percent.toFixed(1)}%`;
                }
            },
            grid: {
                left: '3%',
                right: '15%',
                top: 55,
                bottom: 10,
                containLabel: true
            },
            xAxis: {
                type: 'value',
                max: maxValue,
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
                axisLine: {
                    show: false
                },
                axisTick: {
                    show: false
                }
            },
            series: [
                {
                    name: 'Time',
                    type: 'bar',
                    data: values.map((val, idx) => ({
                        value: val,
                        itemStyle: {
                            color: colors[idx]
                        }
                    })),
                    barWidth: '60%',
                    label: {
                        show: true,
                        position: 'right',
                        formatter: (params) => {
                            const time = params.value;
                            const percent = percentages[categories.length - 1 - params.dataIndex];
                            return `${this.formatTime(time)} (${percent.toFixed(1)}%)`;
                        },
                        fontSize: 10,
                        color: '#555'
                    }
                }
            ]
        };
        
        this.chart.setOption(option, true);
    }
    
    /**
     * Format time value to human-readable string
     */
    formatTime(seconds) {
        if (seconds === undefined || seconds === null || isNaN(seconds)) {
            return '-';
        }
        
        if (seconds < 0.001) {
            return `${(seconds * 1000000).toFixed(0)}us`;
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
     * Dispose of resources
     */
    dispose() {
        // Remove event listeners
        if (this._boundHandlers.onSolveComplete) {
            document.removeEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        }
        if (this._boundHandlers.onSolveStart) {
            document.removeEventListener('fem:jobStarted', this._boundHandlers.onSolveStart);
        }
        
        super.dispose();
    }
}

export default TimingWaterfall;
