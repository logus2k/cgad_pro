/**
 * ServerHardware - Server hardware information display
 * 
 * Shows:
 * - CPU model and cores
 * - RAM
 * - GPU model and memory
 * - CUDA version
 * - Hostname / OS
 * 
 * Type: SYSTEM metric (loads from benchmark API on init)
 * 
 * Location: /src/app/client/script/metrics/widgets/ServerHardware.js
 */

import { BaseMetric } from '../BaseMetric.js';

export class ServerHardware extends BaseMetric {
    constructor(options = {}) {
        super('server-hardware', {
            title: 'Server Hardware',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '200px', right: '780px' },
            ...options
        });
        
        // Data storage
        this.serverConfig = null;
    }
    
    /**
     * Initialize - fetch server config
     */
    init() {
        super.init();
        this.fetchServerConfig();
    }
    
    /**
     * Bind events
     */
    bindEvents() {
        super.bindEvents();
        
        // Refresh on solve complete (in case server changed)
        this._boundHandlers.onSolveComplete = (e) => this.onSolveComplete(e);
        document.addEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
    }
    
    /**
     * Handle solve complete - check for updated server config
     */
    onSolveComplete(event) {
        const data = event.detail;
        
        // If solve_complete includes server_config, use it
        if (data?.server_config) {
            this.serverConfig = data.server_config;
            this.hasData = true;
            this.updateChart();
        }
    }
    
    /**
     * Fetch server config from benchmark API
     */
    async fetchServerConfig() {
        try {
            const apiBase = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
                ? '' 
                : '/fem';
            
            const response = await fetch(`${apiBase}/api/benchmark/server-config`);
            if (!response.ok) {
                console.warn('[ServerHardware] Failed to fetch server config:', response.status);
                return;
            }
            
            const data = await response.json();
            this.serverConfig = data.config || data;
            this.hasData = true;
            this.updateChart();
            
        } catch (error) {
            console.error('[ServerHardware] Error fetching server config:', error);
        }
    }
    
    /**
     * Get default chart option (empty state)
     */
    getDefaultOption() {
        return {
            title: {
                text: 'Loading server info...',
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
     * Update chart with server hardware info
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData || !this.serverConfig) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        const cfg = this.serverConfig;
        
        // Build stats
        const stats = [
            { label: 'Hostname', value: cfg.hostname || '-' },
            { label: 'CPU', value: this.shortenCpuName(cfg.cpu_model) },
            { label: 'CPU Cores', value: cfg.cpu_cores ? `${cfg.cpu_cores} cores` : '-' },
            { label: 'RAM', value: cfg.ram_gb ? `${cfg.ram_gb} GB` : '-' },
            { label: 'GPU', value: this.shortenGpuName(cfg.gpu_model) },
            { label: 'GPU Memory', value: cfg.gpu_memory_gb ? `${cfg.gpu_memory_gb} GB` : '-' },
            { label: 'CUDA', value: cfg.cuda_version || '-' }
        ];
        
        const option = {
            title: {
                text: 'Server Hardware',
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
                    top: 30,
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
        const lineHeight = 28;
        const labelWidth = 85;
        
        return stats.flatMap((stat, idx) => [
            {
                type: 'text',
                top: idx * lineHeight,
                style: {
                    text: stat.label,
                    fontSize: 11,
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
                    fontSize: 11,
                    fill: '#333',
                    fontWeight: 'bold'
                }
            }
        ]);
    }
    
    /**
     * Shorten GPU name for display
     */
    shortenGpuName(name) {
        if (!name) return '-';
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
        if (!name) return '-';
        
        // "13th Gen Intel(R) Core(TM) i9-13900K" -> "Intel i9-13900K"
        let shortened = name
            .replace(/\(R\)/g, '')
            .replace(/\(TM\)/g, '')
            .replace(/13th Gen /g, '')
            .replace(/12th Gen /g, '')
            .replace(/11th Gen /g, '')
            .replace('Intel Core ', 'Intel ')
            .replace('AMD Ryzen ', 'Ryzen ')
            .trim();
        
        // Truncate if still too long
        if (shortened.length > 25) {
            shortened = shortened.substring(0, 22) + '...';
        }
        
        return shortened;
    }
    
    /**
     * Reset to initial state
     */
    reset() {
        // Don't reset server config - it's static
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this._boundHandlers.onSolveComplete) {
            document.removeEventListener('fem:solveComplete', this._boundHandlers.onSolveComplete);
        }
        
        super.dispose();
    }
}

export default ServerHardware;
