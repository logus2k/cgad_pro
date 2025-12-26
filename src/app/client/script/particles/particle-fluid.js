import * as THREE from '../../library/three.module.min.js';
import { ParticleBase } from './particle-base.js';

/**
 * ParticleFluid - Soft sprite-based fluid rendering
 * Uses gaussian sprites with additive blending to create fluid-like appearance
 * Particles visually merge where they overlap
 * Good for: Water, smoke, fog, flowing liquids
 */
export class ParticleFluid extends ParticleBase {
    constructor(group, config = {}) {
        super(group, {
            ...ParticleFluid.getDefaults(),
            ...config
        });
        
        this.spriteTexture = null;
    }
    
    static get typeName() {
        return 'fluid';
    }
    
    static getDefaults() {
        return {
            particleCount: 1000,
            particleSize: 0.06,           // World-space size
            particleColor: 0x3399ff,      // Water blue
            particleOpacity: 0.4,         // Opacity for soft blending
            colorBySpeed: true,
            additiveBlending: true,       // Makes particles merge visually
            sizeAttenuation: true         // Size decreases with distance
        };
    }
    
    /**
     * Create a soft gaussian sprite texture
     */
    createSpriteTexture() {
        const size = 64;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        
        // Create radial gradient (gaussian-like falloff)
        const center = size / 2;
        const gradient = ctx.createRadialGradient(center, center, 0, center, center, center);
        
        // Soft gaussian falloff
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.2, 'rgba(255, 255, 255, 0.7)');
        gradient.addColorStop(0.4, 'rgba(255, 255, 255, 0.4)');
        gradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.15)');
        gradient.addColorStop(0.8, 'rgba(255, 255, 255, 0.05)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        
        return texture;
    }
    
    /**
     * Create point cloud with soft sprites
     */
    create(count) {
        this.dispose();
        
        // Create sprite texture
        this.spriteTexture = this.createSpriteTexture();
        
        const geometry = new THREE.BufferGeometry();
        
        // Position buffer
        const positions = new Float32Array(count * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
        
        // Color buffer
        const colors = new Float32Array(count * 3);
        const baseColor = new THREE.Color(this.config.particleColor);
        for (let i = 0; i < count; i++) {
            colors[i * 3] = baseColor.r;
            colors[i * 3 + 1] = baseColor.g;
            colors[i * 3 + 2] = baseColor.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.attributes.color.setUsage(THREE.DynamicDrawUsage);
        
        // Use standard PointsMaterial with sprite texture
        const material = new THREE.PointsMaterial({
            size: this.config.particleSize,
            map: this.spriteTexture,
            vertexColors: true,
            transparent: true,
            opacity: this.config.particleOpacity,
            blending: this.config.additiveBlending ? THREE.AdditiveBlending : THREE.NormalBlending,
            sizeAttenuation: this.config.sizeAttenuation,
            depthWrite: false
        });
        
        this.mesh = new THREE.Points(geometry, material);
        
        // Apply transform
        this.mesh.scale.set(this.scale, this.scale, this.scale);
        this.mesh.position.copy(this.offset);
        
        this.group.add(this.mesh);
        
        console.log(`ParticleFluid: Created ${count} soft sprite particles`);
    }
    
    /**
     * Update particle positions and colors
     */
    update(positions, velocities, maxSpeed) {
        if (!this.mesh) return;
        
        const posAttr = this.mesh.geometry.attributes.position;
        const colorAttr = this.mesh.geometry.attributes.color;
        
        const count = positions.length / 3;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            // Update position
            posAttr.array[idx] = positions[idx];
            posAttr.array[idx + 1] = positions[idx + 1];
            posAttr.array[idx + 2] = positions[idx + 2];
            
            // Update color based on speed
            if (velocities && this.config.colorBySpeed) {
                const speed = Math.sqrt(
                    velocities[idx] ** 2 + 
                    velocities[idx + 1] ** 2 + 
                    velocities[idx + 2] ** 2
                );
                const normalizedSpeed = maxSpeed > 0 ? speed / maxSpeed : 0.5;
                const color = this.speedToColor(normalizedSpeed);
                colorAttr.array[idx] = color.r;
                colorAttr.array[idx + 1] = color.g;
                colorAttr.array[idx + 2] = color.b;
            }
        }
        
        posAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;
    }
    
    /**
     * Override speedToColor for fluid - use water-like colors
     * Deep blue (slow) -> Cyan -> Light blue/white (fast)
     */
    speedToColor(t) {
        t = Math.max(0, Math.min(1, t));
        let r, g, b;
        
        if (t < 0.33) {
            // Deep blue to cyan
            const s = t / 0.33;
            r = 0.1;
            g = 0.4 + s * 0.3;
            b = 0.7 + s * 0.15;
        } else if (t < 0.66) {
            // Cyan to light blue
            const s = (t - 0.33) / 0.33;
            r = 0.1 + s * 0.4;
            g = 0.7 + s * 0.2;
            b = 0.85 + s * 0.1;
        } else {
            // Light blue to white/foam
            const s = (t - 0.66) / 0.34;
            r = 0.5 + s * 0.5;
            g = 0.9 + s * 0.1;
            b = 0.95 + s * 0.05;
        }
        
        return { r, g, b };
    }
    
    /**
     * Set particle color
     */
    setColor(color) {
        super.setColor(color);
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        const needsRecreate = newConfig.particleCount !== undefined;
        
        Object.assign(this.config, newConfig);
        
        if (!needsRecreate && this.mesh) {
            if (newConfig.particleSize !== undefined) {
                this.mesh.material.size = this.config.particleSize;
            }
            if (newConfig.particleOpacity !== undefined) {
                this.mesh.material.opacity = this.config.particleOpacity;
            }
            if (newConfig.additiveBlending !== undefined) {
                this.mesh.material.blending = this.config.additiveBlending 
                    ? THREE.AdditiveBlending 
                    : THREE.NormalBlending;
            }
        }
        
        return needsRecreate;
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this.spriteTexture) {
            this.spriteTexture.dispose();
            this.spriteTexture = null;
        }
        super.dispose();
    }
}
