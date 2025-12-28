/**
 * ClientHardware - Client browser and system information display
 * 
 * Shows:
 * - Browser name and version
 * - Operating system
 * - Screen resolution
 * - Device memory (if available)
 * - GPU renderer (WebGL)
 * 
 * Type: SYSTEM metric (detects on init)
 * 
 * Location: /src/app/client/script/metrics/widgets/ClientHardware.js
 */

import { BaseMetric } from '../BaseMetric.js';

export class ClientHardware extends BaseMetric {
    constructor(options = {}) {
        super('client-hardware', {
            title: 'Client Hardware',
            defaultWidth: 360,
            defaultHeight: 260,
            position: { top: '200px', right: '400px' },
            ...options
        });
        
        // Data storage
        this.clientConfig = null;
    }
    
    /**
     * Initialize - detect client hardware
     */
    init() {
        super.init();
        this.detectClientHardware();
    }
    
    /**
     * Bind events
     */
    bindEvents() {
        super.bindEvents();
        // No special events needed - client info is static
    }
    
    /**
     * Detect client hardware information
     */
    detectClientHardware() {
        const config = {};
        
        // Browser detection
        const ua = navigator.userAgent;
        config.browser = this.detectBrowser(ua);
        
        // OS detection
        config.os = this.detectOS(ua);
        
        // Screen info
        config.screenResolution = `${window.screen.width} x ${window.screen.height}`;
        config.windowSize = `${window.innerWidth} x ${window.innerHeight}`;
        config.devicePixelRatio = window.devicePixelRatio || 1;
        
        // Device memory (Chrome only)
        if (navigator.deviceMemory) {
            config.deviceMemory = `${navigator.deviceMemory} GB`;
        }
        
        // Hardware concurrency (logical processors)
        if (navigator.hardwareConcurrency) {
            config.cpuCores = `${navigator.hardwareConcurrency} threads`;
        }
        
        // WebGL GPU info
        const gpuInfo = this.detectWebGLGpu();
        config.gpuVendor = gpuInfo.vendor;
        config.gpuRenderer = gpuInfo.renderer;
        
        this.clientConfig = config;
        this.hasData = true;
        this.updateChart();
    }
    
    /**
     * Detect browser from user agent
     */
    detectBrowser(ua) {
        if (ua.includes('Firefox/')) {
            const match = ua.match(/Firefox\/(\d+)/);
            return `Firefox ${match ? match[1] : ''}`;
        }
        if (ua.includes('Edg/')) {
            const match = ua.match(/Edg\/(\d+)/);
            return `Edge ${match ? match[1] : ''}`;
        }
        if (ua.includes('Chrome/')) {
            const match = ua.match(/Chrome\/(\d+)/);
            return `Chrome ${match ? match[1] : ''}`;
        }
        if (ua.includes('Safari/') && !ua.includes('Chrome')) {
            const match = ua.match(/Version\/(\d+)/);
            return `Safari ${match ? match[1] : ''}`;
        }
        return 'Unknown';
    }
    
    /**
     * Detect OS from user agent
     */
    detectOS(ua) {
        if (ua.includes('Windows NT 10')) return 'Windows 10/11';
        if (ua.includes('Windows')) return 'Windows';
        if (ua.includes('Mac OS X')) {
            const match = ua.match(/Mac OS X (\d+[._]\d+)/);
            if (match) {
                return `macOS ${match[1].replace('_', '.')}`;
            }
            return 'macOS';
        }
        if (ua.includes('Linux')) return 'Linux';
        if (ua.includes('Android')) return 'Android';
        if (ua.includes('iOS') || ua.includes('iPhone') || ua.includes('iPad')) return 'iOS';
        return 'Unknown';
    }
    
    /**
     * Detect GPU via WebGL
     */
    detectWebGLGpu() {
        const result = { vendor: '-', renderer: '-' };
        
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (gl) {
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo) {
                    result.vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || '-';
                    result.renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '-';
                    
                    // Shorten renderer name
                    result.renderer = this.shortenRendererName(result.renderer);
                }
            }
        } catch (e) {
            console.warn('[ClientHardware] WebGL detection failed:', e);
        }
        
        return result;
    }
    
    /**
     * Shorten WebGL renderer name
     */
    shortenRendererName(name) {
        if (!name || name === '-') return '-';
        
        // Remove common prefixes/suffixes
        let shortened = name
            .replace('ANGLE (', '')
            .replace(')', '')
            .replace(', Direct3D11', '')
            .replace(', Direct3D9', '')
            .replace(' Direct3D11 vs_5_0 ps_5_0', '')
            .replace('NVIDIA ', '')
            .replace('AMD ', '')
            .replace('Intel(R) ', 'Intel ')
            .trim();
        
        // Truncate if too long
        if (shortened.length > 30) {
            shortened = shortened.substring(0, 27) + '...';
        }
        
        return shortened;
    }
    
    /**
     * Get default chart option (empty state)
     */
    getDefaultOption() {
        return {
            title: {
                text: 'Detecting client info...',
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
     * Update chart with client hardware info
     */
    updateChart() {
        if (!this.chart || this.chart.isDisposed()) {
            return;
        }
        
        if (!this.hasData || !this.clientConfig) {
            this.chart.setOption(this.getDefaultOption(), true);
            return;
        }
        
        const cfg = this.clientConfig;
        
        // Build stats
        const stats = [
            { label: 'Browser', value: cfg.browser || '-' },
            { label: 'OS', value: cfg.os || '-' },
            { label: 'Screen', value: cfg.screenResolution || '-' },
            { label: 'Window', value: cfg.windowSize || '-' },
            { label: 'CPU Threads', value: cfg.cpuCores || '-' },
            { label: 'GPU', value: cfg.gpuRenderer || '-' }
        ];
        
        // Add device memory if available
        if (cfg.deviceMemory) {
            stats.splice(4, 0, { label: 'Memory', value: cfg.deviceMemory });
        }
        
        const option = {
            title: {
                text: 'Client Hardware',
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
     * Reset to initial state
     */
    reset() {
        // Don't reset client config - it's static
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        super.dispose();
    }
}

export default ClientHardware;
