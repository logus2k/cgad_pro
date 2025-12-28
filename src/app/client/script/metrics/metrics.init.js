/**
 * Metrics System Initialization
 * 
 * Entry point for the metrics visualization system.
 * Initializes MetricsManager when the DOM is ready.
 * 
 * This file should be imported in application_start.js or loaded separately.
 * 
 * Location: /src/app/client/script/metrics/metrics.init.js
 */

import { initMetricsManager, getMetricsManager } from './MetricsManager.js';

let initialized = false;

/**
 * Initialize the metrics system
 */
function init() {
    if (initialized) {
        console.warn('[MetricsInit] Already initialized');
        return;
    }
    
    const manager = initMetricsManager();
    
    // Expose to window for debugging
    window.metricsManager = manager;
    
    initialized = true;
    console.log('[MetricsInit] Metrics system ready');
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // DOM already loaded, init immediately
    init();
}

export { init, getMetricsManager };
