/**
 * Metrics Settings Tab Initialization
 * Binds checkboxes to global toggle functions
 */

function initMetricsSettings() {
    const bindings = [
        { id: 'cfg-toggle-grid', fn: (checked) => window.toggleGrid?.(checked) },
        { id: 'cfg-toggle-particles', fn: (checked) => window.toggleParticles?.(checked) },
        { id: 'cfg-toggle-3dextrusion', fn: (checked) => window.toggle3DExtrusion?.(checked) },
        { id: 'cfg-toggle-axis-scale', fn: (checked) => window.toggleAxisScale?.(checked) }
    ];
    
    bindings.forEach(({ id, fn }) => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => fn(e.target.checked));
        }
    });
    
    // Special handling for 3D View toggle (has transition animation)
    const viewCheckbox = document.getElementById('cfg-toggle-view');
    if (viewCheckbox) {
        viewCheckbox.addEventListener('change', (e) => {
            const cameraController = window.cameraController;
            
            // Ignore if currently transitioning
            if (cameraController?.isTransitioning) {
                // Revert checkbox to match actual state
                e.target.checked = !cameraController.is2DMode;
                return;
            }
            
            // Disable checkbox during transition
            viewCheckbox.disabled = true;
            
            window.toggleView?.();
            
            // Re-enable after transition completes
            // Transition duration is configurable, default ~1.5s, add buffer
            const duration = (cameraController?.options?.transitionDuration || 1.5) * 1000 + 100;
            setTimeout(() => {
                viewCheckbox.disabled = false;
            }, duration);
        });
    }
    
    console.log('[MetricsSettings] Initialized');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMetricsSettings);
} else {
    initMetricsSettings();
}

export { initMetricsSettings };
