/**
 * SettingsManager - Handles UI-to-Engine synchronization
 */
export class SettingsManager {
    constructor() {
        this.initEventListeners();
        this.syncInitialState();
    }

    initEventListeners() {
        // --- Simulation Tab ---
        this.bind('cfg-use-particles', 'change', (e) => {
            window.useParticleAnimation = e.target.checked;
            // Logic to toggle particle system in the scene would go here
        });

        this.bind('cfg-particle-density', 'input', (e) => {
            if (window.particleSystem) window.particleSystem.setDensity(parseFloat(e.target.value));
        });

        this.bind('cfg-use-extrusion', 'change', (e) => {
            window.use3DExtrusion = e.target.checked;
        });

        this.bind('cfg-extrusion-style', 'change', (e) => {
            window.extrusionType = e.target.value;
        });

        // --- Global Tab ---
        this.bind('cfg-view-margin', 'input', (e) => {
            if (window.setViewMargin) window.setViewMargin(parseFloat(e.target.value));
        });

        this.bind('cfg-camera-speed', 'input', (e) => {
            if (window.setViewDuration) window.setViewDuration(parseFloat(e.target.value));
        });

        this.bind('cfg-bg-speed', 'input', (e) => {
            if (window.waveBackground) window.waveBackground.setSpeed(parseFloat(e.target.value));
        });

        // Inside your settings.manager.js initEventListeners()
        this.bind('cfg-hud-opacity', 'input', (e) => {
            const val = e.target.value;
            // Target all HUD panels and update their standard opacity
            const panels = document.querySelectorAll('.hud');
            panels.forEach(panel => {
                panel.style.opacity = val;
            });
        });
    }

    bind(id, event, callback) {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, callback);
    }

    syncInitialState() {
        // Sets UI values to match the current global constants on load
        const syncMap = {
            'cfg-use-particles': window.useParticleAnimation,
            'cfg-use-extrusion': window.use3DExtrusion,
            'cfg-extrusion-style': window.extrusionType,
            'cfg-hud-opacity': 0.8
        };

        for (const [id, val] of Object.entries(syncMap)) {
            const el = document.getElementById(id);
            if (el) {
                if (el.type === 'checkbox') el.checked = val;
                else el.value = val;
            }
        }
    }
}
