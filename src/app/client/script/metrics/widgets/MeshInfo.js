/**
 * MeshInfo - Display mesh statistics and complexity indicators
 * 
 * Shows:
 * - Node and element counts
 * - Element type (Quad8)
 * - System size (matrix dimensions)
 * - Estimated non-zeros and memory
 * - Complexity indicator (Small/Medium/Large/Very Large)
 * 
 * Type: EARLY metric (appears on mesh loaded, before solve completes)
 * 
 * Location: /src/app/client/script/metrics/widgets/MeshInfo.js
 */

import { BaseMetric } from '../BaseMetric.js';

// Complexity thresholds based on node count
const COMPLEXITY_LEVELS = [
    { max: 10000, label: 'Small', color: '#67c23a', bars: 2 },
    { max: 100000, label: 'Medium', color: '#e6a23c', bars: 5 },
    { max: 500000, label: 'Large', color: '#f56c6c', bars: 8 },
    { max: Infinity, label: 'Very Large', color: '#911', bars: 10 }
];

// Average non-zeros per row for Quad8 elements (approx 27 for 2D thermal)
const AVG_NONZEROS_PER_ROW = 27;

export class MeshInfo extends BaseMetric {
    constructor(options = {}) {
        super('mesh-stats', {
            title: 'Mesh Info',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '420px', right: '400px' },
            ...options
        });
        
        // Data storage
        this.meshName = null;
        this.nodes = null;
        this.elements = null;
    }
    
    /**
     * Bind events
     */
    bindEvents() {
        super.bindEvents();
        
        // Listen for mesh loaded - show panel with mesh info
        this._boundHandlers.onMeshLoaded = (e) => this.onMeshLoaded(e);
        document.addEventListener('fem:meshLoaded', this._boundHandlers.onMeshLoaded);
        
        // Listen for job started - get mesh name early
        this._boundHandlers.onJobStarted = (e) => this.onJobStarted(e);
        document.addEventListener('fem:jobStarted', this._boundHandlers.onJobStarted);
        
        // Listen for mesh selected from gallery
        this._boundHandlers.onMeshSelected = (e) => this.onMeshSelected(e);
        document.addEventListener('meshSelected', this._boundHandlers.onMeshSelected);
    }
    
    /**
     * Handle mesh selected from gallery (before solve)
     */
    onMeshSelected(event) {
        const detail = event.detail;
        const mesh = detail?.mesh;
        
        if (mesh) {
            this.meshName = mesh.name || mesh.file?.split('/').pop() || 'Unknown';
            this.nodes = mesh.nodes || null;
            this.elements = mesh.elements || null;
            
            console.log(`[MeshInfo] meshSelected: ${this.meshName}, nodes=${this.nodes}, elements=${this.elements}`);
            
            if (this.nodes && this.elements) {
                this.hasData = true;
                this.updateChart();
                
                if (this.isEnabled()) {
                    this.show();
                }
            }
        }
    }
    
    /**
     * Handle job started - extract mesh name
     */
    onJobStarted(event) {
        const data = event.detail;
        if (data?.mesh_file) {
            this.meshName = data.mesh_file.split('/').pop();
        }
    }
    
    /**
     * Handle mesh loaded - update with full info
     */
    onMeshLoaded(event) {
        const data = event.detail;
        
        if (!data) return;
        
        this.nodes = data.nodes || null;
        this.elements = data.elements || null;
        
        // Try to get mesh name from various sources
        if (!this.meshName) {
            this.meshName = data.model_name || data.mesh_file?.split('/').pop() || 'Unknown';
        }
        
        this.hasData = this.nodes !== null && this.elements !== null;
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
                text: 'Awaiting mesh...',
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
     * Update chart with mesh info
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        // Calculate derived metrics
        const systemSize = this.nodes;
        const nonZerosEst = this.nodes * AVG_NONZEROS_PER_ROW;
        const memoryBytes = nonZerosEst * 8; // 8 bytes per float64
        const complexity = this.getComplexity(this.nodes);
        
        // Build rich text display using ECharts graphic elements
        const option = {
            title: {
                text: this.meshName || 'Mesh',
                left: 'center',
                top: 3,
                textStyle: {
                    fontSize: 14,
                    fontWeight: 'bold',
                    color: '#333'
                }
            },
            graphic: [
                // Stats table (including complexity)
                {
                    type: 'group',
                    left: 20,
                    top: 35,
                    children: this.buildStatsGraphic(systemSize, nonZerosEst, memoryBytes, complexity)
                }
            ]
        };
        
        this.chart.setOption(option, true);
    }
    
    /**
     * Build stats display as graphic elements
     */
    buildStatsGraphic(systemSize, nonZerosEst, memoryBytes, complexity) {
        const stats = [
            { label: 'Nodes', value: this.formatNumber(this.nodes) },
            { label: 'Elements', value: this.formatNumber(this.elements) },
            { label: 'Element Type', value: 'Quad8' },
            { label: 'System Size', value: `${this.formatNumber(systemSize)} x ${this.formatNumber(systemSize)}` },
            { label: 'Non-zeros Est.', value: this.formatLargeNumber(nonZerosEst) },
            { label: 'Memory Est.', value: this.formatBytes(memoryBytes) },
            { label: 'Complexity', value: complexity.label, color: complexity.color }
        ];
        
        const lineHeight = 26;
        const labelWidth = 100;
        
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
     * Determine complexity level based on node count
     */
    getComplexity(nodes) {
        for (const level of COMPLEXITY_LEVELS) {
            if (nodes < level.max) {
                return level;
            }
        }
        return COMPLEXITY_LEVELS[COMPLEXITY_LEVELS.length - 1];
    }
    
    /**
     * Format number with thousand separators
     */
    formatNumber(num) {
        if (num === null || num === undefined) return '-';
        return num.toLocaleString();
    }
    
    /**
     * Format large numbers (millions, billions)
     */
    formatLargeNumber(num) {
        if (num === null || num === undefined) return '-';
        
        if (num >= 1e9) {
            return `~${(num / 1e9).toFixed(1)}B`;
        } else if (num >= 1e6) {
            return `~${(num / 1e6).toFixed(1)}M`;
        } else if (num >= 1e3) {
            return `~${(num / 1e3).toFixed(1)}K`;
        }
        return `~${num}`;
    }
    
    /**
     * Format bytes to human readable
     */
    formatBytes(bytes) {
        if (bytes === null || bytes === undefined) return '-';
        
        if (bytes >= 1e9) {
            return `~${(bytes / 1e9).toFixed(1)} GB`;
        } else if (bytes >= 1e6) {
            return `~${(bytes / 1e6).toFixed(0)} MB`;
        } else if (bytes >= 1e3) {
            return `~${(bytes / 1e3).toFixed(0)} KB`;
        }
        return `~${bytes} B`;
    }
    
    /**
     * Reset to initial state
     */
    reset() {
        this.meshName = null;
        this.nodes = null;
        this.elements = null;
        this.hasData = false;
        
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getDefaultOption(), true);
        }
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this._boundHandlers.onMeshLoaded) {
            document.removeEventListener('fem:meshLoaded', this._boundHandlers.onMeshLoaded);
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

export default MeshInfo;
