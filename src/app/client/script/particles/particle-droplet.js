import * as THREE from '../../library/three.module.min.js';
import { ParticleBase } from './particle-base.js';

/**
 * ParticleDroplet - Elongated droplet particles that stretch in flow direction
 * Good for: Water droplets, rain, fast-moving fluids
 */
export class ParticleDroplet extends ParticleBase {
    constructor(group, config = {}) {
        super(group, {
            ...ParticleDroplet.getDefaults(),
            ...config
        });
        
        this.tempMatrix = new THREE.Matrix4();
        this.tempQuaternion = new THREE.Quaternion();
        this.tempScale = new THREE.Vector3();
        this.tempPosition = new THREE.Vector3();
        this.upVector = new THREE.Vector3(1, 0, 0);  // Default flow direction
    }
    
    static get typeName() {
        return 'droplet';
    }
    
    static getDefaults() {
        return {
            particleCount: 600,
            particleSize: 0.012,
            particleColor: 0x3399ff,     // Water blue
            particleOpacity: 0.85,
            particleRoughness: 0.2,
            particleMetalness: 0.0,
            colorBySpeed: false,
            minStretch: 1.0,             // Minimum elongation
            maxStretch: 4.0,             // Maximum elongation at max speed
            dropletSegments: 8           // Geometry detail
        };
    }
    
    /**
     * Create instanced droplet mesh (elongated spheres)
     */
    create(count) {
        this.dispose();
        
        // Create elongated sphere geometry (stretched along Y axis, will be rotated)
        const geometry = new THREE.SphereGeometry(
            this.config.particleSize, 
            this.config.dropletSegments, 
            this.config.dropletSegments
        );
        
        const material = new THREE.MeshStandardMaterial({
            color: this.config.particleColor,
            roughness: this.config.particleRoughness,
            metalness: this.config.particleMetalness,
            transparent: true,
            opacity: this.config.particleOpacity,
            envMapIntensity: 0.8
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
        
        console.log(`ParticleDroplet: Created ${count} droplet particles`);
    }
    
    /**
     * Update particle positions, orientations, and stretch based on velocity
     */
    update(positions, velocities, maxSpeed) {
        if (!this.mesh) return;
        
        const count = positions.length / 3;
        const { minStretch, maxStretch } = this.config;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            // Position
            this.tempPosition.set(
                positions[idx + 0],
                positions[idx + 1],
                positions[idx + 2]
            );
            
            // Get velocity for this particle
            let vx = 1, vy = 0, vz = 0;
            let speed = 1;
            
            if (velocities) {
                vx = velocities[idx + 0];
                vy = velocities[idx + 1];
                vz = velocities[idx + 2];
                speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            }
            
            // Calculate stretch based on speed
            const normalizedSpeed = maxSpeed > 0 ? speed / maxSpeed : 0.5;
            const stretch = minStretch + (maxStretch - minStretch) * normalizedSpeed;
            
            // Scale: stretched along velocity direction
            this.tempScale.set(stretch, 1, 1);
            
            // Rotation: align X-axis with velocity direction
            if (speed > 0.001) {
                const velocityDir = new THREE.Vector3(vx, vy, vz).normalize();
                this.tempQuaternion.setFromUnitVectors(this.upVector, velocityDir);
            } else {
                this.tempQuaternion.identity();
            }
            
            // Compose matrix
            this.tempMatrix.compose(this.tempPosition, this.tempQuaternion, this.tempScale);
            this.mesh.setMatrixAt(i, this.tempMatrix);
            
            // Update color if using speed-based coloring
            if (this.config.colorBySpeed && this.mesh.instanceColor) {
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
        const needsRecreate = super.updateConfig(newConfig);
        
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
