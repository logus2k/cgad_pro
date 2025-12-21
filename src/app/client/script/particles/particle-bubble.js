import * as THREE from '../../library/three.module.min.js';
import { ParticleBase } from './particle-base.js';

/**
 * ParticleBubble - Transparent bubble particles with shiny surface
 * Good for: Air bubbles in water, foam, carbonation
 */
export class ParticleBubble extends ParticleBase {
    constructor(group, config = {}) {
        super(group, {
            ...ParticleBubble.getDefaults(),
            ...config
        });
        
        this.tempMatrix = new THREE.Matrix4();
    }
    
    static get typeName() {
        return 'bubble';
    }
    
    static getDefaults() {
        return {
            particleCount: 500,
            particleSize: 0.02,
            particleColor: 0xffffff,     // White/clear
            particleOpacity: 0.3,
            particleRoughness: 0.0,
            particleMetalness: 0.0,
            colorBySpeed: false,
            rimPower: 2.0,               // Edge glow intensity
            variableSize: true,          // Random size variation
            sizeVariation: 0.5           // Size variation range (0-1)
        };
    }
    
    /**
     * Create instanced bubble mesh
     */
    create(count) {
        this.dispose();
        
        const geometry = new THREE.SphereGeometry(this.config.particleSize, 16, 12);
        
        // Create custom material for bubble effect
        const material = new THREE.MeshPhysicalMaterial({
            color: this.config.particleColor,
            roughness: this.config.particleRoughness,
            metalness: this.config.particleMetalness,
            transparent: true,
            opacity: this.config.particleOpacity,
            transmission: 0.9,           // Glass-like transmission
            thickness: 0.5,              // Refraction thickness
            envMapIntensity: 1.0,
            clearcoat: 1.0,              // Shiny surface coating
            clearcoatRoughness: 0.0,
            ior: 1.33,                   // Index of refraction (water = 1.33)
            side: THREE.DoubleSide
        });
        
        this.mesh = new THREE.InstancedMesh(geometry, material, count);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        
        // Store random size multipliers for each bubble
        this.sizeMultipliers = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            if (this.config.variableSize) {
                const variation = this.config.sizeVariation;
                this.sizeMultipliers[i] = 1 - variation / 2 + Math.random() * variation;
            } else {
                this.sizeMultipliers[i] = 1;
            }
        }
        
        // Create instance color buffer if needed
        if (this.config.colorBySpeed) {
            this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(count * 3), 3
            );
            this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        }
        
        // Apply transform
        this.mesh.scale.set(this.scale, this.scale, this.scale);
        this.mesh.position.copy(this.offset);
        
        this.group.add(this.mesh);
        
        console.log(`ParticleBubble: Created ${count} bubble particles`);
    }
    
    /**
     * Update particle positions with variable sizes
     */
    update(positions, velocities, maxSpeed) {
        if (!this.mesh) return;
        
        const count = positions.length / 3;
        const tempScale = new THREE.Vector3();
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            // Variable size for each bubble
            const size = this.sizeMultipliers[i];
            tempScale.set(size, size, size);
            
            // Compose matrix with position and scale
            this.tempMatrix.makeScale(size, size, size);
            this.tempMatrix.setPosition(
                positions[idx + 0],
                positions[idx + 1],
                positions[idx + 2]
            );
            
            this.mesh.setMatrixAt(i, this.tempMatrix);
            
            // Update color if using speed-based coloring
            if (this.config.colorBySpeed && this.mesh.instanceColor && velocities) {
                const speed = Math.sqrt(
                    velocities[idx + 0] ** 2 + 
                    velocities[idx + 1] ** 2 + 
                    velocities[idx + 2] ** 2
                );
                const normalizedSpeed = maxSpeed > 0 ? speed / maxSpeed : 0.5;
                const color = this.speedToColor(normalizedSpeed);
                this.mesh.instanceColor.setXYZ(i, color.r, color.g, color.b);
            }
        }
        
        this.mesh.instanceMatrix.needsUpdate = true;
        
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
        }
    }
    
    /**
     * Set particle color
     */
    setColor(color) {
        super.setColor(color);
        
        if (this.mesh && this.mesh.material) {
            this.mesh.material.color.setHex(color);
        }
    }
    
    /**
     * Enable/disable speed-based coloring
     */
    setColorBySpeed(enabled) {
        super.setColorBySpeed(enabled);
        
        if (enabled && this.mesh && !this.mesh.instanceColor) {
            const count = this.mesh.count;
            this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(count * 3), 3
            );
            this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        }
        
        if (!enabled && this.mesh) {
            this.mesh.material.color.setHex(this.config.particleColor);
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        // Size and variable size changes require recreate
        const needsRecreate = (
            newConfig.particleCount !== undefined ||
            newConfig.particleSize !== undefined ||
            newConfig.variableSize !== undefined ||
            newConfig.sizeVariation !== undefined
        );
        
        Object.assign(this.config, newConfig);
        
        if (!needsRecreate && this.mesh) {
            if (newConfig.particleColor !== undefined && !this.config.colorBySpeed) {
                this.mesh.material.color.setHex(this.config.particleColor);
            }
            if (newConfig.particleOpacity !== undefined) {
                this.mesh.material.opacity = this.config.particleOpacity;
            }
            if (newConfig.particleRoughness !== undefined) {
                this.mesh.material.roughness = this.config.particleRoughness;
            }
        }
        
        return needsRecreate;
    }
}
