/**
 * Global Z-Index Manager for HUD Panels
 * 
 * Provides a shared z-index counter so that any panel (menu or metric)
 * can be brought to front with a single click, regardless of type.
 * 
 * Usage:
 *   import { getTopZ, bringToFront } from './zIndexManager.js';
 *   
 *   // Get next z-index
 *   panel.style.zIndex = getTopZ();
 *   
 *   // Or bring a panel to front
 *   bringToFront(panelElement);
 * 
 * Location: /src/app/client/script/zIndexManager.js
 */

// Shared z-index counter starting above typical page elements
let globalTopZ = 20;

/**
 * Get the current top z-index and increment the counter
 * @returns {number} The next z-index to use
 */
export function getTopZ() {
    globalTopZ += 1;
    return globalTopZ;
}

/**
 * Get current top z-index without incrementing
 * @returns {number} The current top z-index
 */
export function peekTopZ() {
    return globalTopZ;
}

/**
 * Bring a panel element to the front
 * @param {HTMLElement} panel - The panel element to bring to front
 */
export function bringToFront(panel) {
    if (!panel) return;
    
    const newZ = getTopZ();
    panel.style.zIndex = String(newZ);
    return newZ;
}

/**
 * Set the global z-index to at least the given value
 * Useful when initializing to sync with existing panels
 * @param {number} minZ - Minimum z-index to set
 */
export function ensureMinZ(minZ) {
    if (minZ > globalTopZ) {
        globalTopZ = minZ;
    }
}

/**
 * Scan all HUD panels and sync the counter to the highest existing z-index
 * Call this on initialization to sync with any pre-existing panels
 */
export function syncWithExistingPanels() {
    const panels = document.querySelectorAll('.hud');
    let maxZ = globalTopZ;
    
    panels.forEach(panel => {
        const z = parseInt(panel.style.zIndex) || 0;
        if (z > maxZ) {
            maxZ = z;
        }
    });
    
    globalTopZ = maxZ;
}

// Export for direct access if needed
export default {
    getTopZ,
    peekTopZ,
    bringToFront,
    ensureMinZ,
    syncWithExistingPanels
};
