/**
 * Benchmark Panel Initialization
 * 
 * Integrates BenchmarkPanel with MenuManager visibility system.
 * Initializes the panel lazily when first shown.
 * 
 * Location: /src/app/client/script/benchmark.init.js
 */

import { BenchmarkPanel } from './benchmark.js';

let benchmarkPanel = null;
let initialized = false;

/**
 * Initialize benchmark panel (called once when first shown)
 */
function initBenchmark() {
    if (initialized) return;
    
    const container = document.querySelector('.benchmark-container');
    if (!container) {
        console.error('[Benchmark] Container .benchmark-container not found');
        return;
    }
    
    // Determine API base URL based on current location
    // For production (logus2k.com), API is at /fem
    // For localhost development, API is at root or specify port
    let apiBase = '';
    
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Local development - API on same port or specify
        apiBase = '';
    } else {
        // Production - API proxied through /fem path
        apiBase = '/fem';
    }
    
    console.log('[Benchmark] Using API base:', apiBase || '(root)');
    
    benchmarkPanel = new BenchmarkPanel('.benchmark-container', {
        apiBase: apiBase,
        pollInterval: 10000,  // 10 seconds
        autoRefresh: true
    });
    
    initialized = true;
    
    // Expose to window for debugging
    window.benchmarkPanel = benchmarkPanel;
    
    console.log('[Benchmark] Panel initialized');
}

/**
 * Set up observer to detect when benchmark panel becomes visible
 */
function setupVisibilityObserver() {
    const panel = document.getElementById('hud-benchmark');
    if (!panel) {
        console.warn('[Benchmark] Panel #hud-benchmark not found');
        return;
    }
    
    // Use MutationObserver to detect class changes (visible class)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                if (panel.classList.contains('visible')) {
                    initBenchmark();
                    // Refresh data when panel is shown
                    if (benchmarkPanel) {
                        benchmarkPanel.refresh();
                    }
                }
            }
        });
    });
    
    observer.observe(panel, { attributes: true });
    
    // Also check if already visible (in case initialVisibility was set)
    if (panel.classList.contains('visible')) {
        initBenchmark();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupVisibilityObserver);
} else {
    setupVisibilityObserver();
}

// Export for external access
export { benchmarkPanel, initBenchmark };
