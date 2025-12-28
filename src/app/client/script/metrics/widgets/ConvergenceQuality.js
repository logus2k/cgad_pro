/**
 * ConvergenceQuality - Post-solve convergence analysis metrics
 * 
 * Shows:
 * - Final residual (absolute)
 * - Final relative residual
 * - Total iterations
 * - Convergence rate (average reduction per iteration)
 * - Convergence status (converged/not converged)
 * 
 * Type: POST-SOLVE metric (appears after solve completes)
 * 
 * Location: /src/app/client/script/metrics/widgets/ConvergenceQuality.js
 */

import { BaseMetric } from '../BaseMetric.js';

export class ConvergenceQuality extends BaseMetric {
    constructor(options = {}) {
        super('convergence-quality', {
            title: 'Convergence Quality',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '420px', right: '780px' },
            ...options
        });
        
        // Data storage
        this.converged = null;
        this.iterations = null;
        this.finalResidual = null;
        this.finalRelativeResidual = null;
        this.convergenceRate = null;
        this.solverType = null;
    }
    
    /**
     * Bind events
     */
    bindEvents() {
        super.bindEvents();
        
        // Listen for solve complete
        this._boundHandlers.onSolveComplete = (e) => this.onSolveComplete(e);
        document.addEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        
        // Listen for job started to reset
        this._boundHandlers.onJobStarted = (e) => this.onJobStarted(e);
        document.addEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        
        // Track final residual from progress events
        this._boundHandlers.onSolveProgress = (e) => this.onSolveProgress(e);
        document.addEventListener('fem:solveProgress', this._boundHandlers.onSolveProgress);
    }
    
    /**
     * Handle job started - reset state
     */
    onJobStarted(event) {
        this.reset();
    }
    
    /**
     * Handle solve progress - track residuals for final values
     */
    onSolveProgress(event) {
        const data = event.detail;
        if (!data) return;
        
        // Store latest residual values (will be final when solve completes)
        if (data.residual !== undefined) {
            this.finalResidual = data.residual;
        }
        if (data.relative_residual !== undefined) {
            this.finalRelativeResidual = data.relative_residual;
        }
        if (data.solver_type) {
            this.solverType = data.solver_type;
        }
    }
    
    /**
     * Handle solve complete - display final metrics
     */
    onSolveComplete(event) {
        const data = event.detail;
        if (!data) return;
        
        this.converged = data.converged;
        this.iterations = data.iterations;
        this.solverType = data.solver_type || this.solverType;
        
        // Get final residual from solution_stats if available
        if (data.solution_stats) {
            // Some solvers report final residual in solution_stats
            if (data.solution_stats.final_residual !== undefined) {
                this.finalResidual = data.solution_stats.final_residual;
            }
            if (data.solution_stats.final_relative_residual !== undefined) {
                this.finalRelativeResidual = data.solution_stats.final_relative_residual;
            }
        }
        
        // Calculate convergence rate if we have iterations and residual dropped
        this.calculateConvergenceRate();
        
        this.hasData = true;
        this.updateChart();
        
        if (this.isEnabled()) {
            this.show();
        }
    }
    
    /**
     * Calculate average convergence rate
     * Rate = (final_residual / initial_residual)^(1/iterations)
     * A value close to 1 means slow convergence, close to 0 means fast
     */
    calculateConvergenceRate() {
        // We approximate based on typical initial residual of ~1
        // and the final relative residual
        if (this.finalRelativeResidual && this.iterations && this.iterations > 0) {
            // Convergence rate per iteration (geometric mean)
            this.convergenceRate = Math.pow(this.finalRelativeResidual, 1 / this.iterations);
        } else {
            this.convergenceRate = null;
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
     * Update chart with convergence metrics
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        // Build stats
        const stats = [
            { 
                label: 'Status', 
                value: this.converged ? 'Converged' : 'Not Converged',
                color: this.converged ? '#67c23a' : '#f56c6c'
            },
            { 
                label: 'Iterations', 
                value: this.formatNumber(this.iterations)
            },
            { 
                label: 'Final Residual', 
                value: this.formatExponential(this.finalResidual)
            },
            { 
                label: 'Relative Residual', 
                value: this.formatExponential(this.finalRelativeResidual)
            },
            { 
                label: 'Conv. Rate', 
                value: this.formatConvergenceRate(this.convergenceRate),
                color: this.getConvergenceRateColor(this.convergenceRate)
            },
            {
                label: 'Solver',
                value: this.formatSolverType(this.solverType)
            }
        ];
        
        const option = {
            title: {
                text: this.converged ? 'Solution Converged' : 'Solution Did Not Converge',
                left: 'center',
                top: 3,
                textStyle: {
                    fontSize: 14,
                    fontWeight: 'bold',
                    color: this.converged ? '#67c23a' : '#f56c6c'
                }
            },
            graphic: [
                {
                    type: 'group',
                    left: 20,
                    top: 35,
                    children: this.buildStatsGraphic(stats)
                }
            ]
        };
        
        this.chart.setOption(option, true);
    }
    
    /**
     * Build stats display as graphic elements
     */
    buildStatsGraphic(stats) {
        const lineHeight = 30;
        const labelWidth = 120;
        
        return stats.flatMap((stat, idx) => [
            {
                type: 'text',
                top: idx * lineHeight,
                style: {
                    text: stat.label,
                    fontSize: 12,
                    fill: '#666',
                    fontWeight: 'normal'
                }
            },
            {
                type: 'text',
                top: idx * lineHeight,
                left: labelWidth,
                style: {
                    text: stat.value,
                    fontSize: 12,
                    fill: stat.color || '#333',
                    fontWeight: 'bold'
                }
            }
        ]);
    }
    
    /**
     * Format number with thousand separators
     */
    formatNumber(num) {
        if (num === null || num === undefined) return '-';
        return num.toLocaleString();
    }
    
    /**
     * Format number in exponential notation
     */
    formatExponential(num) {
        if (num === null || num === undefined) return '-';
        return num.toExponential(3);
    }
    
    /**
     * Format convergence rate for display
     */
    formatConvergenceRate(rate) {
        if (rate === null || rate === undefined) return '-';
        
        // Express as percentage reduction per iteration
        const reductionPercent = (1 - rate) * 100;
        
        if (reductionPercent < 0.01) {
            return `${reductionPercent.toExponential(2)}% / iter`;
        } else if (reductionPercent < 1) {
            return `${reductionPercent.toFixed(3)}% / iter`;
        } else {
            return `${reductionPercent.toFixed(2)}% / iter`;
        }
    }
    
    /**
     * Get color for convergence rate (higher reduction = better = greener)
     */
    getConvergenceRateColor(rate) {
        if (rate === null || rate === undefined) return '#333';
        
        const reductionPercent = (1 - rate) * 100;
        
        if (reductionPercent > 1) return '#67c23a';      // Fast (green)
        if (reductionPercent > 0.1) return '#e6a23c';    // Medium (orange)
        return '#f56c6c';                                 // Slow (red)
    }
    
    /**
     * Format solver type for display
     */
    formatSolverType(type) {
        if (!type) return '-';
        
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
        this.converged = null;
        this.iterations = null;
        this.finalResidual = null;
        this.finalRelativeResidual = null;
        this.convergenceRate = null;
        this.solverType = null;
        this.hasData = false;
        
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getDefaultOption(), true);
        }
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this._boundHandlers.onSolveComplete) {
            document.removeEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        }
        if (this._boundHandlers.onJobStarted) {
            document.removeEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        }
        if (this._boundHandlers.onSolveProgress) {
            document.removeEventListener('fem:solveProgress', this._boundHandlers.onSolveProgress);
        }
        
        super.dispose();
    }
}

export default ConvergenceQuality;
