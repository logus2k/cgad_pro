/**
 * Report Workspace Initialization
 * 
 * Integrates ReportWorkspace with MenuManager visibility system.
 * Initializes the workspace lazily when first shown.
 * 
 * Location: /src/app/client/script/report.init.js
 */

import { ReportWorkspace } from './report_workspace.js';

let reportWorkspace = null;
let initialized = false;

/**
 * Initialize report workspace (called once when first shown)
 */
function initReport() {
    if (initialized) return;
    
    const container = document.querySelector('.report-workspace-container');
    if (!container) {
        console.error('[Report] Container .report-workspace-container not found');
        return;
    }
    
    // Determine API base URL based on current location
    let apiBase = '';
    
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        apiBase = '';
    } else {
        apiBase = '/fem';
    }
    
    console.log('[Report] Using API base:', apiBase || '(root)');
    
    reportWorkspace = new ReportWorkspace('.report-workspace-container', {
        apiBase: apiBase
    });
    
    initialized = true;
    
    // Expose to window for debugging
    window.reportWorkspace = reportWorkspace;
    
    console.log('[Report] Workspace initialized');
}

/**
 * Set up observer to detect when report panel becomes visible
 */
function setupVisibilityObserver() {
    const panel = document.getElementById('hud-report');
    if (!panel) {
        console.warn('[Report] Panel #hud-report not found');
        return;
    }
    
    // Use MutationObserver to detect class changes (visible class)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                if (panel.classList.contains('visible')) {
                    initReport();
                    // Refresh data when panel is shown
                    if (reportWorkspace) {
                        reportWorkspace.refresh();
                    }
                }
            }
        });
    });
    
    observer.observe(panel, { attributes: true });
    
    // Also check if already visible (in case initialVisibility was set)
    if (panel.classList.contains('visible')) {
        initReport();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupVisibilityObserver);
} else {
    setupVisibilityObserver();
}

// Export for external access
export { reportWorkspace, initReport };
