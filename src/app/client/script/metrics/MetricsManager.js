/**
 * MetricsManager - Central orchestrator for all metric panels
 * 
 * Responsibilities:
 * - Register and manage metric instances
 * - React to Catalog configuration changes
 * - Coordinate metric lifecycle (init, show, hide, dispose)
 * 
 * Location: /src/app/client/script/metrics/MetricsManager.js
 */

import { TimingWaterfall } from './widgets/TimingWaterfall.js';
import { ConvergencePlot } from './widgets/ConvergencePlot.js';
import { SpeedupFactors } from './widgets/SpeedupFactors.js';
import { MeshInfo } from './widgets/MeshInfo.js';
// Future imports:
// import { ProgressRing } from './widgets/ProgressRing.js';

export class MetricsManager {
    constructor() {
        this.metrics = new Map();
        this.initialized = false;
    }
    
    /**
     * Initialize all metrics
     */
    init() {
        if (this.initialized) {
            console.warn('[MetricsManager] Already initialized');
            return;
        }
        
        // Register all metrics
        this.registerMetrics();
        
        // Initialize each metric
        this.metrics.forEach((metric, id) => {
            try {
                metric.init();
            } catch (error) {
                console.error(`[MetricsManager] Failed to init metric ${id}:`, error);
            }
        });
        
        // Listen for catalog changes
        this.bindEvents();
        
        this.initialized = true;
        console.log(`[MetricsManager] Initialized with ${this.metrics.size} metrics`);
    }
    
    /**
     * Register all metric instances
     */
    registerMetrics() {
        // Performance metrics
        this.register(new TimingWaterfall());
        this.register(new ConvergencePlot());
        this.register(new SpeedupFactors());
        this.register(new MeshInfo());
        
        // Future metrics (uncomment as implemented):
        // this.register(new ProgressRing());
        // this.register(new SolverComparison());
        // this.register(new ConvergenceQuality());
        // this.register(new SolutionRange());
        // this.register(new VelocityStats());
    }
    
    /**
     * Register a metric instance
     * @param {BaseMetric} metric 
     */
    register(metric) {
        if (!metric || !metric.id) {
            console.error('[MetricsManager] Invalid metric', metric);
            return;
        }
        
        if (this.metrics.has(metric.id)) {
            console.warn(`[MetricsManager] Metric ${metric.id} already registered`);
            return;
        }
        
        this.metrics.set(metric.id, metric);
    }
    
    /**
     * Get a metric by ID
     * @param {string} id 
     * @returns {BaseMetric|undefined}
     */
    get(id) {
        return this.metrics.get(id);
    }
    
    /**
     * Bind global event listeners
     */
    bindEvents() {
        // Listen for catalog configuration changes
        document.addEventListener('metricsConfigChanged', (e) => {
            this.onConfigChanged(e.detail?.enabledMetrics || []);
        });
    }
    
    /**
     * Handle catalog configuration changes
     * @param {string[]} enabledMetrics - Array of enabled metric IDs
     */
    onConfigChanged(enabledMetrics) {
        console.log('[MetricsManager] Config changed:', enabledMetrics);
        
        this.metrics.forEach((metric, id) => {
            const shouldBeVisible = enabledMetrics.includes(id);
            
            if (shouldBeVisible && metric.hasData) {
                metric.show();
            } else if (!shouldBeVisible) {
                metric.hide();
            }
        });
    }
    
    /**
     * Show all metrics that are enabled and have data
     */
    showEnabled() {
        const enabledMetrics = this.getEnabledMetricIds();
        
        this.metrics.forEach((metric, id) => {
            if (enabledMetrics.includes(id) && metric.hasData) {
                metric.show();
            }
        });
    }
    
    /**
     * Hide all metrics
     */
    hideAll() {
        this.metrics.forEach((metric) => {
            metric.hide();
        });
    }
    
    /**
     * Reset all metrics
     */
    resetAll() {
        this.metrics.forEach((metric) => {
            metric.reset();
            metric.hide();
        });
    }
    
    /**
     * Get list of enabled metric IDs
     * @returns {string[]}
     */
    getEnabledMetricIds() {
        // Check if metricsCatalog is available
        if (window.metricsCatalog) {
            return window.metricsCatalog.getEnabledMetrics();
        }
        
        // Catalog not initialized yet - return defaults
        return [
            'progress-ring',
            'convergence-plot', 
            'stage-timeline',
            'residual-display',
            'solution-range',
            'velocity-stats',
            'convergence-quality',
            'timing-waterfall',
            'solver-comparison',
            'speedup-factors',
            'mesh-stats'
        ];
    }
    
    /**
     * Dispose all metrics and cleanup
     */
    dispose() {
        this.metrics.forEach((metric) => {
            try {
                metric.dispose();
            } catch (error) {
                console.error(`[MetricsManager] Failed to dispose metric ${metric.id}:`, error);
            }
        });
        
        this.metrics.clear();
        this.initialized = false;
        
        console.log('[MetricsManager] Disposed');
    }
}

// Singleton instance
let instance = null;

/**
 * Get the MetricsManager singleton
 * @returns {MetricsManager}
 */
export function getMetricsManager() {
    if (!instance) {
        instance = new MetricsManager();
    }
    return instance;
}

/**
 * Initialize the MetricsManager
 */
export function initMetricsManager() {
    const manager = getMetricsManager();
    manager.init();
    return manager;
}

export default MetricsManager;
