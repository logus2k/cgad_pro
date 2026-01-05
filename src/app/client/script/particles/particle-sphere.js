import * as THREE from '../../library/three.module.min.js';
import { ParticleBase } from './particle-base.js';

/**
 * ParticleSphere - Solid sphere particles with lighting
 * Good for: Visualizing distinct particles, ping-pong balls, beads
 */
export class ParticleSphere extends ParticleBase {
    constructor(group, config = {}) {
        super(group, {
            ...ParticleSphere.getDefaults(),
            ...config
        });
        
        this.tempMatrix = new THREE.Matrix4();
    }
    
    static get typeName() {
        return 'sphere';
    }
    
    static getDefaults() {
        return {
            particleCount: 800,
            particleSize: 0.015,
            particleColor: 0x222222,
            particleRoughness: 0.4,
            particleMetalness: 0.1,
            colorBySpeed: false
        };
    }
    
    /**
     * Create instanced sphere mesh
     */
    create(count) {
        this.dispose();
        
        const geometry = new THREE.SphereGeometry(this.config.particleSize, 12, 8);
        
        // When colorBySpeed is enabled, use white base color so speed colors show properly
        // (instanceColor multiplies with material color)
        const materialColor = this.config.colorBySpeed ? 0xffffff : this.config.particleColor;
        
        const material = new THREE.MeshStandardMaterial({
            color: materialColor,
            roughness: this.config.particleRoughness,
            metalness: this.config.particleMetalness,
            envMapIntensity: 0.5
        });
        
        this.mesh = new THREE.InstancedMesh(geometry, material, count);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        
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
        
        console.log(`ParticleSphere: Created ${count} sphere particles (colorBySpeed: ${this.config.colorBySpeed})`);
    }
    
    /**
     * Update particle positions and colors
     */
    update(positions, velocities, maxSpeed) {
        if (!this.mesh) return;
        
        const count = positions.length / 3;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            // Update position
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
        
        // Update material color based on colorBySpeed setting
        if (this.mesh && this.mesh.material) {
            if (enabled) {
                // Use white so speed colors show through
                this.mesh.material.color.setHex(0xffffff);
            } else {
                // Restore original particle color
                this.mesh.material.color.setHex(this.config.particleColor);
            }
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        // Check if colorBySpeed is changing
        const colorBySpeedChanging = newConfig.colorBySpeed !== undefined && 
                                      newConfig.colorBySpeed !== this.config.colorBySpeed;
        
        const needsRecreate = super.updateConfig(newConfig);
        
        // Handle colorBySpeed toggle without recreation
        if (colorBySpeedChanging && !needsRecreate) {
            this.setColorBySpeed(this.config.colorBySpeed);
        }
        
        if (!needsRecreate && this.mesh) {
            if (newConfig.particleColor !== undefined && !this.config.colorBySpeed) {
                this.mesh.material.color.setHex(this.config.particleColor);
            }
            if (newConfig.particleRoughness !== undefined) {
                this.mesh.material.roughness = this.config.particleRoughness;
            }
            if (newConfig.particleMetalness !== undefined) {
                this.mesh.material.metalness = this.config.particleMetalness;
            }
        }
        
        return needsRecreate;
    }
}
