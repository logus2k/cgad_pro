/**
 * Metrics Settings Tab Initialization
 * Binds checkboxes to global toggle functions
 */

function initMetricsSettings() {
    const bindings = [
        { id: 'cfg-toggle-grid', fn: (checked) => window.toggleGrid?.(checked) },
        { id: 'cfg-toggle-particles', fn: (checked) => window.toggleParticles?.(checked) },
        { id: 'cfg-toggle-view', fn: (checked) => window.toggleView?.(checked) },
        { id: 'cfg-toggle-3dextrusion', fn: (checked) => window.toggle3DExtrusion?.(checked) }
    ];
    
    bindings.forEach(({ id, fn }) => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => fn(e.target.checked));
        }
    });
    
    console.log('[MetricsSettings] Initialized');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMetricsSettings);
} else {
    initMetricsSettings();
}

export { initMetricsSettings };
