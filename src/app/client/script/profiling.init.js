/**
 * Profiling Panel Initialization
 * 
 * Initializes the ProfilingView when the profiling panel becomes visible.
 * Connects to the shared Socket.IO instance from the FEM client.
 * 
 * Location: /src/app/client/script/profiling.init.js
 */

import { ProfilingView } from './profiling/profiling.view.js';

let profilingView = null;

/**
 * Initialize the profiling view.
 * Called when the panel is first shown.
 */
function initProfiling() {
    if (profilingView) {
        // Already initialized, just refresh
        profilingView.loadSessions();
        return;
    }

    // Get Socket.IO instance from global femClient (set in application_start.js)
    const socket = window.femClient?.socket || null;
    
    if (!socket) {
        console.warn('[ProfilingInit] Socket.IO not available, real-time updates disabled');
    }

    // Initialize the profiling view
    profilingView = new ProfilingView('profiling-container', socket);

    console.log('[ProfilingInit] Profiling panel initialized');
}

// Initialize when panel becomes visible
// Using MutationObserver to detect when panel is shown
const profilingPanel = document.getElementById('hud-profiling');

if (profilingPanel) {
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                if (profilingPanel.classList.contains('visible')) {
                    initProfiling();
                }
            }
        }
    });

    observer.observe(profilingPanel, { attributes: true });

    // Also initialize if panel is already visible
    if (profilingPanel.classList.contains('visible')) {
        initProfiling();
    }
}

// Export for manual initialization if needed
window.initProfiling = initProfiling;
