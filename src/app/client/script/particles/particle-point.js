import * as THREE from '../../library/three.module.min.js';
import { ParticleBase } from './particle-base.js';

/**
 * ParticlePoint - GL Point particles
 * Good for: Dense fluids, water spray, high particle counts, performance
 */
export class ParticlePoint extends ParticleBase {
    constructor(group, config = {}) {
        super(group, {
            ...ParticlePoint.getDefaults(),
            ...config
        });
    }
    
    static get typeName() {
        return 'point';
    }
    
    static getDefaults() {
        return {
            particleCount: 3000,
            particleSize: 0.8,           // Screen-space size
            particleColor: 0x4488ff,     // Light blue (water)
            particleOpacity: 0.7,
            colorBySpeed: true,
            additiveBlending: true,
            sizeAttenuation: true
        };
    }
    
    /**
     * Create point cloud
     */
    create(count) {
        this.dispose();
        
        const geometry = new THREE.BufferGeometry();
        
        // Position buffer
        const positions = new Float32Array(count * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
        
        // Color buffer
        const colors = new Float32Array(count * 3);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.attributes.color.setUsage(THREE.DynamicDrawUsage);
        
        // Size buffer (for variable sizes based on speed)
        const sizes = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            sizes[i] = this.config.particleSize;
        }
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.attributes.size.setUsage(THREE.DynamicDrawUsage);
        
        const material = new THREE.PointsMaterial({
            size: this.config.particleSize,
            vertexColors: true,
            transparent: true,
            opacity: this.config.particleOpacity,
            sizeAttenuation: this.config.sizeAttenuation,
            blending: this.config.additiveBlending ? THREE.AdditiveBlending : THREE.NormalBlending,
            depthWrite: !this.config.additiveBlending
        });
        
        this.mesh = new THREE.Points(geometry, material);
        
        // Apply transform
        this.mesh.scale.set(this.scale, this.scale, this.scale);
        this.mesh.position.copy(this.offset);
        
        this.group.add(this.mesh);
        
        console.log(`ParticlePoint: Created ${count} point particles`);
    }
    
    /**
     * Update particle positions and colors
     */
    update(positions, velocities, maxSpeed) {
        if (!this.mesh) return;
        
        const posAttr = this.mesh.geometry.attributes.position;
        const colorAttr = this.mesh.geometry.attributes.color;
        const sizeAttr = this.mesh.geometry.attributes.size;
        
        const count = positions.length / 3;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            // Update position
            posAttr.array[idx + 0] = positions[idx + 0];
            posAttr.array[idx + 1] = positions[idx + 1];
            posAttr.array[idx + 2] = positions[idx + 2];
            
            // Update color and size based on speed
            if (velocities) {
                const speed = Math.sqrt(
                    velocities[idx + 0] ** 2 + 
                    velocities[idx + 1] ** 2 + 
                    velocities[idx + 2] ** 2
                );
                const normalizedSpeed = maxSpeed > 0 ? speed / maxSpeed : 0.5;
                
                if (this.config.colorBySpeed) {
                    const color = this.speedToColor(normalizedSpeed);
                    colorAttr.array[idx + 0] = color.r;
                    colorAttr.array[idx + 1] = color.g;
                    colorAttr.array[idx + 2] = color.b;
                } else {
                    const c = new THREE.Color(this.config.particleColor);
                    colorAttr.array[idx + 0] = c.r;
                    colorAttr.array[idx + 1] = c.g;
                    colorAttr.array[idx + 2] = c.b;
                }
                
                // Size varies with speed
                sizeAttr.array[i] = this.config.particleSize * (0.5 + normalizedSpeed * 0.8);
            }
        }
        
        posAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;
        sizeAttr.needsUpdate = true;
    }
    
    /**
     * Set particle color
     */
    setColor(color) {
        super.setColor(color);
        // Colors are set per-vertex in update()
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        const needsRecreate = super.updateConfig(newConfig);
        
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
                this.mesh.material.depthWrite = !this.config.additiveBlending;
            }
        }
        
        return needsRecreate;
    }
}
