/**
 * Metrics Settings Tab Initialization
 * Binds checkboxes and sliders to global toggle/config functions
 * 
 * Location: /src/app/client/script/metrics-settings.init.js
 */

function initMetricsSettings() {
    // Checkbox bindings
    const checkboxBindings = [
        { id: 'cfg-toggle-grid', fn: (checked) => window.toggleGrid?.(checked) },
        { id: 'cfg-toggle-particles', fn: (checked) => window.toggleParticles?.(checked) },
        { id: 'cfg-toggle-3dextrusion', fn: (checked) => window.toggle3DExtrusion?.(checked) },
        { id: 'cfg-toggle-axis-scale', fn: (checked) => window.toggleAxisScale?.(checked) }
    ];
    
    checkboxBindings.forEach(({ id, fn }) => {
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
            const duration = (cameraController?.options?.transitionDuration || 1.5) * 1000 + 100;
            setTimeout(() => {
                viewCheckbox.disabled = false;
            }, duration);
        });
    }
    
    // Slider bindings
    initSliderBindings();
    
    console.log('[MetricsSettings] Initialized');
}

/**
 * Initialize slider controls for mesh and particle settings
 */
function initSliderBindings() {
    // Mesh Opacity slider
    const opacitySlider = document.getElementById('cfg-mesh-opacity');
    const opacityValue = document.getElementById('cfg-mesh-opacity-value');
    
    if (opacitySlider) {
        opacitySlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (opacityValue) {
                opacityValue.textContent = `${Math.round(value * 100)}%`;
            }
            window.setOpacity?.(value);
        });
    }
    
    // Particle Count slider
    const particleCountSlider = document.getElementById('cfg-particle-count');
    const particleCountValue = document.getElementById('cfg-particle-count-value');
    
    if (particleCountSlider) {
        particleCountSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value, 10);
            if (particleCountValue) {
                particleCountValue.textContent = value.toLocaleString();
            }
        });
        
        // Apply on change (release) to avoid performance issues during drag
        particleCountSlider.addEventListener('change', (e) => {
            const value = parseInt(e.target.value, 10);
            window.updateParticles?.({ particleCount: value });
        });
    }
    
    // Speed Scale slider
    const speedScaleSlider = document.getElementById('cfg-speed-scale');
    const speedScaleValue = document.getElementById('cfg-speed-scale-value');
    
    if (speedScaleSlider) {
        speedScaleSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (speedScaleValue) {
                speedScaleValue.textContent = `${value.toFixed(1)}x`;
            }
            window.updateParticles?.({ speedScale: value });
        });
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMetricsSettings);
} else {
    initMetricsSettings();
}

export { initMetricsSettings };
