/**
 * SolutionRange - Solution field statistics
 * 
 * Shows:
 * - Minimum value
 * - Maximum value
 * - Mean value
 * - Standard deviation
 * - Range (max - min)
 * 
 * Type: POST-SOLVE metric (appears after solve completes)
 * 
 * Location: /src/app/client/script/metrics/widgets/SolutionRange.js
 */

import { BaseMetric } from '../BaseMetric.js';

export class SolutionRange extends BaseMetric {
    constructor(options = {}) {
        super('solution-range', {
            title: 'Solution Range',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '500px', right: '20px' },
            ...options
        });
        
        // Data storage
        this.uMin = null;
        this.uMax = null;
        this.uMean = null;
        this.uStd = null;
        this.meshName = null;
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
        
        // Listen for mesh selected to get mesh name
        this._boundHandlers.onMeshSelected = (e) => this.onMeshSelected(e);
        document.addEventListener('meshSelected', this._boundHandlers.onMeshSelected);
    }
    
    /**
     * Handle mesh selected - get mesh name
     */
    onMeshSelected(event) {
        const detail = event.detail;
        const mesh = detail?.mesh;
        if (mesh) {
            this.meshName = mesh.name || mesh.file?.split('/').pop() || null;
        }
    }
    
    /**
     * Handle job started - reset state
     */
    onJobStarted(event) {
        this.reset();
    }
    
    /**
     * Handle solve complete - extract solution stats
     */
    onSolveComplete(event) {
        const data = event.detail;
        if (!data) return;
        
        const stats = data.solution_stats;
        if (!stats) {
            console.warn('[SolutionRange] No solution_stats in solve_complete data');
            return;
        }
        
        // Extract values - u_range is [min, max]
        if (stats.u_range && Array.isArray(stats.u_range)) {
            this.uMin = stats.u_range[0];
            this.uMax = stats.u_range[1];
        }
        this.uMean = stats.u_mean;
        this.uStd = stats.u_std;
        
        this.hasData = this.uMin !== null && this.uMax !== null;
        this.updateChart();
        
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
                text: 'Awaiting solution...',
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
     * Update chart with solution stats
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        // Calculate range
        const range = this.uMax - this.uMin;
        
        // Build stats
        const stats = [
            { label: 'Minimum', value: this.formatValue(this.uMin) },
            { label: 'Maximum', value: this.formatValue(this.uMax) },
            { label: 'Range', value: this.formatValue(range) },
            { label: 'Mean', value: this.formatValue(this.uMean) },
            { label: 'Std Deviation', value: this.formatValue(this.uStd) }
        ];
        
        // Title with mesh name if available
        const titleText = this.meshName 
            ? `Solution: ${this.meshName}`
            : 'Solution Statistics';
        
        const option = {
            title: {
                text: titleText,
                left: 'center',
                top: 3,
                textStyle: {
                    fontSize: 14,
                    fontWeight: 'bold',
                    color: '#333'
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
        const lineHeight = 32;
        const labelWidth = 110;
        
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
     * Format numeric value for display
     */
    formatValue(num) {
        if (num === null || num === undefined) return '-';
        
        // Use fixed notation for reasonable ranges, exponential for extreme values
        const abs = Math.abs(num);
        if (abs === 0) {
            return '0.000';
        } else if (abs < 0.001 || abs >= 10000) {
            return num.toExponential(3);
        } else if (abs < 1) {
            return num.toFixed(4);
        } else {
            return num.toFixed(3);
        }
    }
    
    /**
     * Reset to initial state
     */
    reset() {
        this.uMin = null;
        this.uMax = null;
        this.uMean = null;
        this.uStd = null;
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
        if (this._boundHandlers.onMeshSelected) {
            document.removeEventListener('meshSelected', this._boundHandlers.onMeshSelected);
        }
        
        super.dispose();
    }
}

export default SolutionRange;
