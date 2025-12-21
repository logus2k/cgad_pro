import * as THREE from '../../library/three.module.min.js';

/**
 * ParticleBase - Abstract base class for particle visualizations
 * Defines the interface that all particle types must implement
 */
export class ParticleBase {
    constructor(group, config) {
        this.group = group;
        this.config = config;
        this.mesh = null;
        this.scale = 1;
        this.offset = new THREE.Vector3();
    }
    
    /**
     * Get the type name of this particle renderer
     * @returns {string}
     */
    static get typeName() {
        return 'base';
    }
    
    /**
     * Get default configuration for this particle type
     * @returns {Object}
     */
    static getDefaults() {
        return {
            particleCount: 800,
            particleSize: 0.015,
            particleColor: 0x222222,
            colorBySpeed: false
        };
    }
    
    /**
     * Create the Three.js objects for this particle type
     * Must be implemented by subclasses
     * @param {number} count - Number of particles
     */
    create(count) {
        throw new Error('create() must be implemented by subclass');
    }
    
    /**
     * Update particle positions and colors
     * @param {Float32Array} positions - Particle positions [x,y,z, x,y,z, ...]
     * @param {Float32Array} velocities - Particle velocities for stretching/coloring
     * @param {number} maxSpeed - Maximum speed for normalization
     */
    update(positions, velocities, maxSpeed) {
        throw new Error('update() must be implemented by subclass');
    }
    
    /**
     * Set the transform to match the tube mesh
     * @param {number} scale - Scale factor
     * @param {THREE.Vector3} offset - Position offset
     */
    setTransform(scale, offset) {
        this.scale = scale;
        this.offset.copy(offset);
        
        if (this.mesh) {
            this.mesh.scale.set(scale, scale, scale);
            this.mesh.position.copy(offset);
        }
    }
    
    /**
     * Set visibility
     * @param {boolean} visible
     */
    setVisible(visible) {
        if (this.mesh) {
            this.mesh.visible = visible;
        }
    }
    
    /**
     * Set particle color (uniform)
     * @param {number} color - Hex color
     */
    setColor(color) {
        this.config.particleColor = color;
        // Subclasses should override to update material
    }
    
    /**
     * Enable/disable speed-based coloring
     * @param {boolean} enabled
     */
    setColorBySpeed(enabled) {
        this.config.colorBySpeed = enabled;
    }
    
    /**
     * Update configuration
     * @param {Object} newConfig
     * @returns {boolean} - True if particles need to be recreated
     */
    updateConfig(newConfig) {
        const needsRecreate = (
            newConfig.particleCount !== undefined ||
            newConfig.particleSize !== undefined
        );
        
        Object.assign(this.config, newConfig);
        
        return needsRecreate;
    }
    
    /**
     * Convert speed to color using heat map
     * Blue (slow) -> Cyan -> Green -> Yellow -> Red (fast)
     * @param {number} t - Normalized speed [0, 1]
     * @returns {{r: number, g: number, b: number}}
     */
    speedToColor(t) {
        t = Math.max(0, Math.min(1, t));
        let r, g, b;
        
        if (t < 0.25) {
            const s = t / 0.25;
            r = 0; g = s; b = 1;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0; g = 1; b = 1 - s;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = s; g = 1; b = 0;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1; g = 1 - s; b = 0;
        }
        
        return { r, g, b };
    }
    
    /**
     * Dispose of Three.js resources
     */
    dispose() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            
            if (this.mesh.geometry) {
                this.mesh.geometry.dispose();
            }
            
            if (this.mesh.material) {
                if (Array.isArray(this.mesh.material)) {
                    this.mesh.material.forEach(m => m.dispose());
                } else {
                    this.mesh.material.dispose();
                }
            }
            
            this.mesh = null;
        }
    }
}
