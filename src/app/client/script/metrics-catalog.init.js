/**
 * Metrics Catalog Initialization
 * 
 * Initializes the MetricsCatalog component when the Catalog tab is shown.
 * 
 * Location: /src/app/client/script/metrics-catalog.init.js
 */

import { MetricsCatalog, METRICS_CATALOG } from './metrics-catalog.js';

let metricsCatalog = null;
let initialized = false;

/**
 * Initialize metrics catalog
 */
function initCatalog() {
    if (initialized) return;
    
    const container = document.querySelector('.metrics-catalog-container');
    if (!container) {
        console.error('[MetricsCatalog] Container not found');
        return;
    }
    
    metricsCatalog = new MetricsCatalog(container, {
        onChange: (enabledMetrics) => {
            console.log('[MetricsCatalog] Configuration changed:', enabledMetrics);
            // Future: Update Runtime tab to show/hide metrics
        }
    });
    
    initialized = true;
    
    // Expose to window for debugging
    window.metricsCatalog = metricsCatalog;
    window.METRICS_CATALOG = METRICS_CATALOG;
    
    console.log('[MetricsCatalog] Initialized');
}

/**
 * Set up observer for tab switching
 */
function setupTabObserver() {
    // Initialize when Catalog tab is clicked
    document.addEventListener('click', (e) => {
        const tab = e.target.closest('.tab[data-tab="tab-metrics-catalog"]');
        if (tab) {
            // Small delay to allow tab content to become visible
            setTimeout(initCatalog, 50);
        }
    });
    
    // Also check if already on Catalog tab
    const catalogContent = document.getElementById('tab-metrics-catalog-content');
    if (catalogContent && catalogContent.classList.contains('active')) {
        initCatalog();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupTabObserver);
} else {
    setupTabObserver();
}

export { metricsCatalog, METRICS_CATALOG };
