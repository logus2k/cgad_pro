import * as THREE from '../library/three.module.min.js';

/**
 * ParticleFlow - Animated particle flow visualization for tubes
 * Uses instanced spheres for better visibility and spatial grid for fast velocity lookup
 */
export class ParticleFlow {
    constructor(meshExtruder, velocityData, renderCallback, config = {}) {
        this.meshExtruder = meshExtruder;
        this.velocityData = velocityData;
        this.renderCallback = renderCallback;
        
        this.config = {
            particleCount: 800,
            particleRadius: 0.015,
            particleColor: 0x222222,
            particleRoughness: 0.4,
            particleMetalness: 0.1,
            speedScale: 0.5,
            particleMaxLife: 10.0,
            colorBySpeed: false,
            gridResolution: 100,
            inflowDistance: 0.15,
            outflowDistance: 0.25,
            tubeWallMargin: 0.02,  // Fixed margin from tube wall (in world units)
            ...config
        };
        
        this.group = new THREE.Group();
        this.instancedMesh = null;
        this.isAnimating = false;
        this.animationId = null;
        this.lastTime = 0;
        
        // Get bounds from mesh extruder
        this.originalBounds = { ...meshExtruder.originalBounds };
        this.bounds = {
            xMin: this.originalBounds.xMin - this.config.inflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin),
            xMax: this.originalBounds.xMax + this.config.outflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin),
            yMin: this.originalBounds.yMin,
            yMax: this.originalBounds.yMax,
            zMin: meshExtruder.bounds.zMin,
            zMax: meshExtruder.bounds.zMax
        };
        
        // Build spatial grid for fast velocity lookup
        this.buildVelocityGrid();
        
        // Compute scale to match mesh extruder
        this.scale = 50 / Math.max(
            this.originalBounds.xMax - this.originalBounds.xMin,
            this.originalBounds.yMax - this.originalBounds.yMin
        );
        
        // Particle state arrays
        this.particlePositions = null;
        this.particleLifetimes = null;
        this.tempMatrix = new THREE.Matrix4();
        this.tempColor = new THREE.Color();
        
        meshExtruder.group.add(this.group);
        
        console.log('ParticleFlow initialized');
    }
    
    /**
     * Build spatial grid for O(1) velocity lookup
     */
    buildVelocityGrid() {
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        const res = this.config.gridResolution;
        
        console.log(`   Building velocity grid: ${res}x${res}...`);
        
        this.velocityGrid = [];
        this.gridRes = res;
        this.gridDx = (xMax - xMin) / res;
        this.gridDy = (yMax - yMin) / res;
        
        for (let i = 0; i <= res; i++) {
            this.velocityGrid[i] = [];
            for (let j = 0; j <= res; j++) {
                this.velocityGrid[i][j] = { vx: 0, vy: 0, speed: 0, count: 0 };
            }
        }
        
        this.maxSpeed = 0;
        for (let i = 0; i < this.velocityData.abs_vel.length; i++) {
            if (this.velocityData.abs_vel[i] > this.maxSpeed) {
                this.maxSpeed = this.velocityData.abs_vel[i];
            }
        }
        
        const coords = this.meshExtruder.meshData.coordinates;
        const conn = this.meshExtruder.meshData.connectivity;
        
        for (let e = 0; e < conn.length; e++) {
            let cx = 0, cy = 0;
            for (let i = 0; i < 8; i += 2) {
                const nodeId = conn[e][i];
                cx += coords.x[nodeId];
                cy += coords.y[nodeId];
            }
            cx /= 4;
            cy /= 4;
            
            const gi = Math.floor((cx - xMin) / this.gridDx);
            const gj = Math.floor((cy - yMin) / this.gridDy);
            
            if (gi >= 0 && gi <= res && gj >= 0 && gj <= res) {
                const vel = this.velocityData.vel[e];
                const cell = this.velocityGrid[gi][gj];
                cell.vx += vel[0];
                cell.vy += vel[1];
                cell.speed += this.velocityData.abs_vel[e];
                cell.count++;
            }
        }
        
        for (let i = 0; i <= res; i++) {
            for (let j = 0; j <= res; j++) {
                const cell = this.velocityGrid[i][j];
                if (cell.count > 0) {
                    cell.vx /= cell.count;
                    cell.vy /= cell.count;
                    cell.speed /= cell.count;
                }
            }
        }
        
        console.log(`   Velocity grid built, max speed: ${this.maxSpeed.toFixed(3)}`);
    }
    
    /**
     * Get velocity at position using spatial grid (O(1))
     */
    getVelocityAt(x, y) {
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        
        const clampedX = Math.max(xMin, Math.min(xMax, x));
        const clampedY = Math.max(yMin, Math.min(yMax, y));
        
        const gi = Math.floor((clampedX - xMin) / this.gridDx);
        const gj = Math.floor((clampedY - yMin) / this.gridDy);
        
        if (gi >= 0 && gi < this.gridRes && gj >= 0 && gj < this.gridRes) {
            const cell = this.velocityGrid[gi][gj];
            if (cell.count > 0) {
                return { vx: cell.vx, vy: cell.vy, speed: cell.speed };
            }
        }
        
        return { vx: this.maxSpeed * 0.5, vy: 0, speed: this.maxSpeed * 0.5 };
    }
    
    /**
     * Get tube segment at X position, finding the best match for a given Y,Z position
     * Uses radial distance to find which tube the particle is actually in
     */
    getSegmentAtPosition(x, y, z = 0) {
        const { xMin: origXMin, xMax: origXMax } = this.originalBounds;
        
        // Clamp X to original bounds for segment lookup
        const sampleX = Math.max(origXMin, Math.min(origXMax, x));
        const segments = this.meshExtruder.getYSegmentsAtXCached(sampleX);
        
        if (!segments || segments.length === 0) {
            return null;
        }
        
        // If only one segment, return it
        if (segments.length === 1) {
            return segments[0];
        }
        
        // Multiple segments (branches) - find which one the particle is in/closest to
        // Use radial distance from each tube's centerline
        let bestSeg = segments[0];
        let bestRadialDist = Infinity;
        
        for (const seg of segments) {
            const dy = y - seg.centerY;
            const radialDist = Math.sqrt(dy * dy + z * z);
            
            // Prefer segment where particle is inside (radialDist < radius)
            // If inside multiple or none, use closest
            if (radialDist < seg.radius) {
                // Particle is inside this segment
                if (radialDist < bestRadialDist || bestRadialDist >= bestSeg.radius) {
                    bestSeg = seg;
                    bestRadialDist = radialDist;
                }
            } else if (bestRadialDist >= bestSeg.radius) {
                // Not inside any segment yet, track closest
                if (radialDist < bestRadialDist) {
                    bestSeg = seg;
                    bestRadialDist = radialDist;
                }
            }
        }
        
        return bestSeg;
    }
    
    /**
     * Constrain position to stay inside tube cross-section
     * Accounts for particle radius so spheres don't clip through walls
     */
    constrainToTube(x, y, z) {
        const seg = this.getSegmentAtPosition(x, y, z);
        
        if (!seg) {
            return { y, z, constrained: false };
        }
        
        const dy = y - seg.centerY;
        const radialDist = Math.sqrt(dy * dy + z * z);
        
        // Max radius = tube radius - fixed wall margin - particle radius
        const maxRadius = seg.radius - this.config.tubeWallMargin - this.config.particleRadius;
        
        if (radialDist > maxRadius && radialDist > 0.0001) {
            // Push particle back inside
            const scale = maxRadius / radialDist;
            return {
                y: seg.centerY + dy * scale,
                z: z * scale,
                constrained: true
            };
        }
        
        return { y, z, constrained: false };
    }
    
    /**
     * Create instanced mesh for particles (spheres)
     */
    createParticles() {
        if (this.instancedMesh) {
            this.group.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
        }
        
        const count = this.config.particleCount;
        console.log(`Creating ${count} sphere particles...`);
        
        const geometry = new THREE.SphereGeometry(this.config.particleRadius, 12, 8);
        
        const material = new THREE.MeshStandardMaterial({
            color: this.config.particleColor,
            roughness: this.config.particleRoughness,
            metalness: this.config.particleMetalness,
            envMapIntensity: 0.5
        });
        
        this.instancedMesh = new THREE.InstancedMesh(geometry, material, count);
        this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        
        if (this.config.colorBySpeed) {
            this.instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(count * 3), 3
            );
            this.instancedMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        }
        
        this.particlePositions = new Float32Array(count * 3);
        this.particleLifetimes = new Float32Array(count);
        
        for (let i = 0; i < count; i++) {
            this.respawnParticle(i, true);
        }
        
        this.updateInstanceMatrices();
        
        this.instancedMesh.scale.set(this.scale, this.scale, this.scale);
        const { xMin, xMax, yMin } = this.originalBounds;
        const centerX = (xMin + xMax) / 2;
        this.instancedMesh.position.set(-centerX * this.scale, -yMin * this.scale, 0);
        
        this.group.add(this.instancedMesh);
        
        console.log('Sphere particle system created');
    }
    
    /**
     * Respawn a particle at inlet or random position
     */
    respawnParticle(index, randomPosition = false) {
        const { xMin: origXMin, xMax: origXMax } = this.originalBounds;
        const { xMin: extXMin } = this.bounds;
        
        let x, y, z;
        let attempts = 0;
        
        do {
            if (randomPosition && Math.random() < 0.3) {
                // 30% random along tube (for initial distribution)
                x = origXMin + (origXMax - origXMin) * Math.random();
            } else {
                // 70% spawn before inlet (in the extended region)
                x = extXMin + (origXMin - extXMin) * Math.random();
            }
            
            // Get tube cross-section
            const seg = this.getSegmentAtPosition(x, (this.originalBounds.yMin + this.originalBounds.yMax) / 2, 0);
            
            if (seg) {
                // Random position in circular cross-section
                // Account for wall margin and particle radius
                const angle = Math.random() * Math.PI * 2;
                const maxR = seg.radius - this.config.tubeWallMargin - this.config.particleRadius;
                const r = Math.sqrt(Math.random()) * Math.max(0.001, maxR);
                
                y = seg.centerY + Math.cos(angle) * r;
                z = Math.sin(angle) * r;
                break;
            }
            attempts++;
        } while (attempts < 20);
        
        if (attempts >= 20) {
            x = extXMin;
            y = (this.originalBounds.yMin + this.originalBounds.yMax) / 2;
            z = 0;
        }
        
        // For spawns inside the tube (not at inlet), pick correct branch
        if (x >= origXMin) {
            const segments = this.meshExtruder.getYSegmentsAtXCached(x);
            if (segments && segments.length > 1) {
                // Multiple branches - pick one randomly
                const seg = segments[Math.floor(Math.random() * segments.length)];
                const angle = Math.random() * Math.PI * 2;
                const maxR = seg.radius - this.config.tubeWallMargin - this.config.particleRadius;
                const r = Math.sqrt(Math.random()) * Math.max(0.001, maxR);
                y = seg.centerY + Math.cos(angle) * r;
                z = Math.sin(angle) * r;
            }
        }
        
        const idx = index * 3;
        this.particlePositions[idx + 0] = x;
        this.particlePositions[idx + 1] = y;
        this.particlePositions[idx + 2] = z;
        
        this.particleLifetimes[index] = Math.random() * this.config.particleMaxLife;
        
        if (this.config.colorBySpeed && this.instancedMesh.instanceColor) {
            const vel = this.getVelocityAt(x, y);
            const normalizedSpeed = this.maxSpeed > 0 ? vel.speed / this.maxSpeed : 0.5;
            const color = this.speedToColor(normalizedSpeed);
            this.instancedMesh.instanceColor.setXYZ(index, color.r, color.g, color.b);
        }
    }
    
    /**
     * Update all instance matrices from positions
     */
    updateInstanceMatrices() {
        const count = this.config.particleCount;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            this.tempMatrix.setPosition(
                this.particlePositions[idx + 0],
                this.particlePositions[idx + 1],
                this.particlePositions[idx + 2]
            );
            this.instancedMesh.setMatrixAt(i, this.tempMatrix);
        }
        
        this.instancedMesh.instanceMatrix.needsUpdate = true;
        
        if (this.instancedMesh.instanceColor) {
            this.instancedMesh.instanceColor.needsUpdate = true;
        }
    }
    
    /**
     * Color by speed: blue (slow) -> cyan -> green -> yellow -> red (fast)
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
     * Check if particle is in valid flow region (extended bounds)
     */
    isInFlowRegion(x, y, z) {
        // Get segment for this position (using Y and Z for proper branch detection)
        const seg = this.getSegmentAtPosition(x, y, z);
        if (!seg) return false;
        
        // Check radial distance
        const dy = y - seg.centerY;
        const radialDist = Math.sqrt(dy * dy + z * z);
        
        // Allow slightly outside (just wall margin, not particle radius)
        const maxRadius = seg.radius - this.config.tubeWallMargin * 0.5;
        
        return radialDist < maxRadius;
    }
    
    /**
     * Update particles
     */
    update(deltaTime) {
        if (!this.instancedMesh) return;
        
        const { xMax: extXMax } = this.bounds;
        const speedScale = this.config.speedScale;
        const count = this.config.particleCount;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            let x = this.particlePositions[idx + 0];
            let y = this.particlePositions[idx + 1];
            let z = this.particlePositions[idx + 2];
            
            // Get velocity at current position
            const vel = this.getVelocityAt(x, y);
            
            // Move particle
            x += vel.vx * speedScale * deltaTime;
            y += vel.vy * speedScale * deltaTime;
            
            // CONSTRAIN to tube cross-section (prevents escaping)
            const constrained = this.constrainToTube(x, y, z);
            y = constrained.y;
            z = constrained.z;
            
            // Update lifetime
            this.particleLifetimes[i] -= deltaTime;
            
            // Check respawn conditions
            const pastExtendedOutlet = x > extXMax;
            const expired = this.particleLifetimes[i] <= 0;
            
            // Also check if completely outside flow region (shouldn't happen now with constraint)
            const outsideFlow = !this.isInFlowRegion(x, y, z);
            
            if (pastExtendedOutlet || expired || outsideFlow) {
                this.respawnParticle(i, false);
            } else {
                this.particlePositions[idx + 0] = x;
                this.particlePositions[idx + 1] = y;
                this.particlePositions[idx + 2] = z;
                
                if (this.config.colorBySpeed && this.instancedMesh.instanceColor) {
                    const normalizedSpeed = this.maxSpeed > 0 ? vel.speed / this.maxSpeed : 0.5;
                    const color = this.speedToColor(normalizedSpeed);
                    this.instancedMesh.instanceColor.setXYZ(i, color.r, color.g, color.b);
                }
            }
        }
        
        this.updateInstanceMatrices();
    }
    
    /**
     * Start animation
     */
    start() {
        if (this.isAnimating) return;
        
        if (!this.instancedMesh) {
            this.createParticles();
        }
        
        this.isAnimating = true;
        this.lastTime = performance.now();
        this.instancedMesh.visible = true;
        
        console.log('Particle animation started');
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            const now = performance.now();
            const deltaTime = Math.min((now - this.lastTime) / 1000, 0.1);
            this.lastTime = now;
            
            this.update(deltaTime);
            
            if (this.renderCallback) {
                this.renderCallback();
            }
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    /**
     * Stop animation
     */
    stop() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        console.log('Particle animation stopped');
    }
    
    /**
     * Toggle visibility
     */
    setVisible(visible) {
        if (this.instancedMesh) {
            this.instancedMesh.visible = visible;
        }
        
        if (visible && !this.isAnimating) {
            this.start();
        } else if (!visible && this.isAnimating) {
            this.stop();
        }
    }
    
    /**
     * Set particle color (uniform)
     */
    setColor(color) {
        this.config.particleColor = color;
        this.config.colorBySpeed = false;
        
        if (this.instancedMesh) {
            this.instancedMesh.material.color.setHex(color);
        }
    }
    
    /**
     * Enable/disable speed-based coloring
     */
    setColorBySpeed(enabled) {
        this.config.colorBySpeed = enabled;
        
        if (enabled && this.instancedMesh && !this.instancedMesh.instanceColor) {
            const count = this.config.particleCount;
            this.instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(count * 3), 3
            );
            this.instancedMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        }
        
        if (!enabled && this.instancedMesh) {
            this.instancedMesh.material.color.setHex(this.config.particleColor);
        }
    }
    
    /**
     * Update config and recreate particles
     */
    updateConfig(newConfig) {
        const needsRecreate = 
            newConfig.particleCount !== undefined ||
            newConfig.particleRadius !== undefined;
        
        Object.assign(this.config, newConfig);
        
        if (newConfig.inflowDistance !== undefined || newConfig.outflowDistance !== undefined) {
            this.bounds.xMin = this.originalBounds.xMin - this.config.inflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin);
            this.bounds.xMax = this.originalBounds.xMax + this.config.outflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin);
        }
        
        if (needsRecreate) {
            const wasAnimating = this.isAnimating;
            if (wasAnimating) this.stop();
            this.createParticles();
            if (wasAnimating) this.start();
        } else {
            if (this.instancedMesh) {
                if (newConfig.particleColor !== undefined && !this.config.colorBySpeed) {
                    this.instancedMesh.material.color.setHex(this.config.particleColor);
                }
                if (newConfig.particleRoughness !== undefined) {
                    this.instancedMesh.material.roughness = this.config.particleRoughness;
                }
                if (newConfig.particleMetalness !== undefined) {
                    this.instancedMesh.material.metalness = this.config.particleMetalness;
                }
            }
        }
    }
    
    /**
     * Dispose
     */
    dispose() {
        this.stop();
        
        if (this.instancedMesh) {
            this.group.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
        }
        
        this.meshExtruder.group.remove(this.group);
    }
}
