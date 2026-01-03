/**
 * Report Workspace Initialization
 * 
 * Integrates ReportWorkspace with MenuManager visibility system.
 * Initializes the workspace lazily when first shown.
 * Configures marked.js with KaTeX and highlight.js extensions.
 * 
 * Location: /src/app/client/script/report.init.js
 */

import { ReportWorkspace } from './report_workspace.js';

let reportWorkspace = null;
let initialized = false;
let markedConfigured = false;

/**
 * Configure marked.js with KaTeX math and highlight.js code highlighting
 */
function configureMarked() {
    if (markedConfigured) return;
    
    if (!window.marked) {
        console.warn('[Report] marked.js not loaded');
        return;
    }
    
    // Configure highlight.js extension for code syntax highlighting
    // Must be configured BEFORE KaTeX extension
    if (window.markedHighlight && window.hljs) {
        window.marked.use(window.markedHighlight.markedHighlight({
            langPrefix: 'hljs language-',
            highlight: function(code, lang) {
                if (lang && window.hljs.getLanguage(lang)) {
                    try {
                        return window.hljs.highlight(code, { language: lang }).value;
                    } catch (e) {
                        console.warn('[Report] Highlight error:', e);
                    }
                }
                // Fallback to auto-detection
                try {
                    return window.hljs.highlightAuto(code).value;
                } catch (e) {
                    return code;
                }
            }
        }));
        console.log('[Report] highlight.js extension configured');
    }
    
    // Configure KaTeX extension for math rendering
    if (window.markedKatex) {
        window.marked.use(window.markedKatex({
            throwOnError: false,
            output: 'html'
        }));
        console.log('[Report] KaTeX extension configured');
    }
    
    markedConfigured = true;
}

/**
 * Initialize report workspace (called once when first shown)
 */
function initReport() {
    if (initialized) return;
    
    // Configure marked extensions first
    configureMarked();
    
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
    document.addEventListener('DOMContentLoaded', () => {
        configureMarked();
        setupVisibilityObserver();
    });
} else {
    configureMarked();
    setupVisibilityObserver();
}

// Export for external access
export { reportWorkspace, initReport, configureMarked };
