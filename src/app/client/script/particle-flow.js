import * as THREE from '../library/three.module.min.js';
import { ParticleSphere } from './particles/particle-sphere.js';
import { ParticlePoint } from './particles/particle-point.js';
import { ParticleDroplet } from './particles/particle-droplet.js';
import { ParticleBubble } from './particles/particle-bubble.js';

/**
 * Available particle type classes
 */
const PARTICLE_TYPES = {
    sphere: ParticleSphere,
    point: ParticlePoint,
    droplet: ParticleDroplet,
    bubble: ParticleBubble
};

/**
 * ParticleFlow - Main controller for particle flow visualization
 * Manages velocity field, particle positions, and delegates rendering to particle type classes
 */
export class ParticleFlow {
    constructor(meshExtruder, velocityData, renderCallback, config = {}) {
        this.meshExtruder = meshExtruder;
        this.velocityData = velocityData;
        this.renderCallback = renderCallback;
        
        // Default particle type
        const particleType = config.particleType || 'sphere';
        const ParticleClass = PARTICLE_TYPES[particleType];
        
        this.config = {
            particleType: particleType,
            particleCount: 800,
            speedScale: 0.5,
            particleMaxLife: 10.0,
            gridResolution: 100,
            inflowDistance: 0.15,
            outflowDistance: 0.25,
            tubeWallMargin: 0.02,
            // Extrusion mode: 'cylindrical' (radial constraint) or 'rectangular' (Y bounds constraint)
            extrusionMode: 'cylindrical',
            // Merge type-specific defaults
            ...ParticleClass.getDefaults(),
            ...config
        };
        
        this.group = new THREE.Group();
        this.particleRenderer = null;
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
        
        // Debug: compare with mesh extruder scale
        console.log('ParticleFlow scale:', this.scale.toFixed(4));
        console.log('MeshExtruder scale:', meshExtruder.scale?.toFixed(4) || 'not set yet');
        
        // Compute offset - must match mesh extruder's fitMeshToView
        const { xMin, xMax, yMin } = this.originalBounds;
        const { zMin, zMax } = this.bounds;
        const centerX = (xMin + xMax) / 2;
        const centerZ = (zMin + zMax) / 2;
        this.offset = new THREE.Vector3(-centerX * this.scale, -yMin * this.scale, -centerZ * this.scale);
        
        // Particle state arrays
        this.particlePositions = null;
        this.particleVelocities = null;
        this.particleLifetimes = null;
        
        meshExtruder.group.add(this.group);
        
        console.log('ParticleFlow initialized');
    }
    
    /**
     * Get list of available particle types
     */
    static getAvailableTypes() {
        return Object.keys(PARTICLE_TYPES);
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
        
        // Clamp Y to mesh bounds for grid lookup
        const clampedY = Math.max(yMin, Math.min(yMax, y));
        const gj = Math.floor((clampedY - yMin) / this.gridDy);
        const gjClamped = Math.max(0, Math.min(this.gridRes - 1, gj));
        
        // Before inlet: use velocity at inlet (x = xMin)
        if (x < xMin) {
            const gi = 0;
            const cell = this.velocityGrid[gi][gjClamped];
            if (cell.count > 0) {
                return { vx: cell.vx, vy: cell.vy, speed: cell.speed };
            }
            // Fallback: move toward inlet with modest speed
            return { vx: this.maxSpeed * 0.3, vy: 0, speed: this.maxSpeed * 0.3 };
        }
        
        // After outlet: use velocity at outlet (x = xMax)
        if (x > xMax) {
            const gi = this.gridRes - 1;
            const cell = this.velocityGrid[gi][gjClamped];
            if (cell.count > 0) {
                return { vx: cell.vx, vy: cell.vy, speed: cell.speed };
            }
            // Fallback: continue with modest speed
            return { vx: this.maxSpeed * 0.3, vy: 0, speed: this.maxSpeed * 0.3 };
        }
        
        // Inside mesh bounds: use grid velocity
        const gi = Math.floor((x - xMin) / this.gridDx);
        const giClamped = Math.max(0, Math.min(this.gridRes - 1, gi));
        
        const cell = this.velocityGrid[giClamped][gjClamped];
        if (cell.count > 0) {
            return { vx: cell.vx, vy: cell.vy, speed: cell.speed };
        }
        
        // Empty grid cell inside bounds - interpolate from neighbors or use fallback
        return { vx: this.maxSpeed * 0.3, vy: 0, speed: this.maxSpeed * 0.3 };
    }
    
    /**
     * Get tube segment at X position, finding the best match for a given Y,Z position
     * Cylindrical mode: uses radial distance to segment center
     * Rectangular mode: uses Y bounds containment
     */
    getSegmentAtPosition(x, y, z = 0) {
        const { xMin: origXMin, xMax: origXMax } = this.originalBounds;
        
        const sampleX = Math.max(origXMin, Math.min(origXMax, x));
        const segments = this.meshExtruder.getYSegmentsAtXCached(sampleX);
        
        if (!segments || segments.length === 0) {
            return null;
        }
        
        if (segments.length === 1) {
            return segments[0];
        }
        
        // Rectangular mode: find segment containing Y position
        if (this.config.extrusionMode === 'rectangular') {
            // First, try to find a segment that contains the Y position
            for (const seg of segments) {
                if (y >= seg.yMin && y <= seg.yMax) {
                    return seg;
                }
            }
            
            // If not inside any segment, find the closest one by Y distance
            let bestSeg = segments[0];
            let bestDist = Infinity;
            
            for (const seg of segments) {
                const dist = y < seg.yMin ? (seg.yMin - y) : (y - seg.yMax);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestSeg = seg;
                }
            }
            
            return bestSeg;
        }
        
        // Cylindrical mode: use radial distance (original behavior)
        let bestSeg = segments[0];
        let bestRadialDist = Infinity;
        
        for (const seg of segments) {
            const dy = y - seg.centerY;
            const radialDist = Math.sqrt(dy * dy + z * z);
            
            if (radialDist < seg.radius) {
                if (radialDist < bestRadialDist || bestRadialDist >= bestSeg.radius) {
                    bestSeg = seg;
                    bestRadialDist = radialDist;
                }
            } else if (bestRadialDist >= bestSeg.radius) {
                if (radialDist < bestRadialDist) {
                    bestSeg = seg;
                    bestRadialDist = radialDist;
                }
            }
        }
        
        return bestSeg;
    }
    
    /**
     * Constrain position to stay inside tube/slab cross-section
     * Cylindrical mode: radial constraint (circular cross-section)
     * Rectangular mode: Y bounds constraint (rectangular cross-section)
     */
    constrainToTube(x, y, z) {
        const seg = this.getSegmentAtPosition(x, y, z);
        
        if (!seg) {
            return { y, z, constrained: false };
        }
        
        const particleSize = this.config.particleSize || 0.015;
        const margin = this.config.tubeWallMargin + particleSize;
        
        // Rectangular mode: constrain Y to segment bounds, Z to slab bounds
        if (this.config.extrusionMode === 'rectangular') {
            const { zMin, zMax } = this.meshExtruder.bounds;
            
            let constrained = false;
            let newY = y;
            let newZ = z;
            
            // Constrain Y to THIS segment's bounds (important for bifurcation)
            const yMinBound = seg.yMin + margin;
            const yMaxBound = seg.yMax - margin;
            
            if (y < yMinBound) {
                newY = yMinBound;
                constrained = true;
            } else if (y > yMaxBound) {
                newY = yMaxBound;
                constrained = true;
            }
            
            // Constrain Z to slab bounds
            const zMinBound = zMin + margin;
            const zMaxBound = zMax - margin;
            
            if (z < zMinBound) {
                newZ = zMinBound;
                constrained = true;
            } else if (z > zMaxBound) {
                newZ = zMaxBound;
                constrained = true;
            }
            
            return { y: newY, z: newZ, constrained };
        }
        
        // Cylindrical mode: radial constraint (original behavior)
        const dy = y - seg.centerY;
        const radialDist = Math.sqrt(dy * dy + z * z);
        const maxRadius = seg.radius - margin;
        
        if (radialDist > maxRadius && radialDist > 0.0001) {
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
     * Check if particle is in valid flow region
     */
    isInFlowRegion(x, y, z) {
        const seg = this.getSegmentAtPosition(x, y, z);
        if (!seg) return false;
        
        const margin = this.config.tubeWallMargin * 0.5;
        
        // Rectangular mode: check Y and Z bounds
        if (this.config.extrusionMode === 'rectangular') {
            const { zMin, zMax } = this.meshExtruder.bounds;
            
            const inYBounds = y >= seg.yMin + margin && y <= seg.yMax - margin;
            const inZBounds = z >= zMin + margin && z <= zMax - margin;
            
            return inYBounds && inZBounds;
        }
        
        // Cylindrical mode: check radial distance
        const dy = y - seg.centerY;
        const radialDist = Math.sqrt(dy * dy + z * z);
        const maxRadius = seg.radius - margin;
        
        return radialDist < maxRadius;
    }
    
    /**
     * Initialize particle positions
     */
    initializeParticles() {
        const count = this.config.particleCount;
        
        this.particlePositions = new Float32Array(count * 3);
        this.particleVelocities = new Float32Array(count * 3);
        this.particleLifetimes = new Float32Array(count);
        
        for (let i = 0; i < count; i++) {
            this.respawnParticle(i, true);
        }
    }
    
    /**
     * Respawn a particle
     */
    respawnParticle(index, randomPosition = false) {
        const { xMin: origXMin, xMax: origXMax } = this.originalBounds;
        const { xMin: extXMin, zMin, zMax } = this.bounds;
        const particleSize = this.config.particleSize || 0.015;
        const margin = this.config.tubeWallMargin + particleSize;
        
        let x, y, z;
        let attempts = 0;
        
        do {
            if (randomPosition && Math.random() < 0.3) {
                x = origXMin + (origXMax - origXMin) * Math.random();
            } else {
                x = extXMin + (origXMin - extXMin) * Math.random();
            }
            
            const seg = this.getSegmentAtPosition(x, (this.originalBounds.yMin + this.originalBounds.yMax) / 2, 0);
            
            if (seg) {
                // Rectangular mode: uniform distribution in Y and Z bounds
                if (this.config.extrusionMode === 'rectangular') {
                    const yRange = (seg.yMax - margin) - (seg.yMin + margin);
                    const zRange = (zMax - margin) - (zMin + margin);
                    
                    y = seg.yMin + margin + Math.random() * yRange;
                    z = zMin + margin + Math.random() * zRange;
                } 
                // Cylindrical mode: circular distribution
                else {
                    const angle = Math.random() * Math.PI * 2;
                    const maxR = seg.radius - margin;
                    const r = Math.sqrt(Math.random()) * Math.max(0.001, maxR);
                    
                    y = seg.centerY + Math.cos(angle) * r;
                    z = Math.sin(angle) * r;
                }
                break;
            }
            attempts++;
        } while (attempts < 20);
        
        if (attempts >= 20) {
            x = extXMin;
            y = (this.originalBounds.yMin + this.originalBounds.yMax) / 2;
            z = 0;
        }
        
        // Handle multiple branches
        if (x >= origXMin) {
            const segments = this.meshExtruder.getYSegmentsAtXCached(x);
            if (segments && segments.length > 1) {
                const seg = segments[Math.floor(Math.random() * segments.length)];
                
                // Rectangular mode: uniform distribution in Y and Z bounds
                if (this.config.extrusionMode === 'rectangular') {
                    const yRange = (seg.yMax - margin) - (seg.yMin + margin);
                    const zRange = (zMax - margin) - (zMin + margin);
                    
                    y = seg.yMin + margin + Math.random() * yRange;
                    z = zMin + margin + Math.random() * zRange;
                }
                // Cylindrical mode: circular distribution
                else {
                    const angle = Math.random() * Math.PI * 2;
                    const maxR = seg.radius - margin;
                    const r = Math.sqrt(Math.random()) * Math.max(0.001, maxR);
                    y = seg.centerY + Math.cos(angle) * r;
                    z = Math.sin(angle) * r;
                }
            }
        }
        
        const idx = index * 3;
        this.particlePositions[idx + 0] = x;
        this.particlePositions[idx + 1] = y;
        this.particlePositions[idx + 2] = z;
        
        // Store velocity
        const vel = this.getVelocityAt(x, y);
        this.particleVelocities[idx + 0] = vel.vx;
        this.particleVelocities[idx + 1] = vel.vy;
        this.particleVelocities[idx + 2] = 0;
        
        this.particleLifetimes[index] = Math.random() * this.config.particleMaxLife;
    }
    
    /**
     * Create particle renderer of specified type
     */
    createParticleRenderer(type = null) {
        // Dispose existing renderer
        if (this.particleRenderer) {
            this.particleRenderer.dispose();
        }
        
        const particleType = type || this.config.particleType;
        const ParticleClass = PARTICLE_TYPES[particleType];
        
        if (!ParticleClass) {
            console.error(`Unknown particle type: ${particleType}`);
            return;
        }
        
        this.config.particleType = particleType;
        
        // Create new renderer
        this.particleRenderer = new ParticleClass(this.group, this.config);
        this.particleRenderer.setTransform(this.scale, this.offset);
        this.particleRenderer.create(this.config.particleCount);
        
        console.log(`Particle type set to: ${particleType}`);
    }
    
    /**
     * Set particle type
     */
    setParticleType(type) {
        if (!PARTICLE_TYPES[type]) {
            console.error(`Unknown particle type: ${type}. Available: ${Object.keys(PARTICLE_TYPES).join(', ')}`);
            return;
        }
        
        const wasAnimating = this.isAnimating;
        if (wasAnimating) this.stop();
        
        // Get defaults for new type
        const ParticleClass = PARTICLE_TYPES[type];
        const typeDefaults = ParticleClass.getDefaults();
        
        // Keep only flow-related settings, use new type's visual defaults
        this.config = {
            // Flow settings (keep these)
            particleCount: this.config.particleCount,
            speedScale: this.config.speedScale,
            particleMaxLife: this.config.particleMaxLife,
            gridResolution: this.config.gridResolution,
            inflowDistance: this.config.inflowDistance,
            outflowDistance: this.config.outflowDistance,
            tubeWallMargin: this.config.tubeWallMargin,
            // Type-specific visual settings (use new type's defaults)
            ...typeDefaults,
            // Set the type
            particleType: type
        };
        
        // Reinitialize particles with new count if different
        if (typeDefaults.particleCount && typeDefaults.particleCount !== this.particlePositions?.length / 3) {
            this.config.particleCount = typeDefaults.particleCount;
            this.initializeParticles();
        }
        
        this.createParticleRenderer(type);
        
        if (wasAnimating) this.start();
    }
    
    /**
     * Update particle simulation
     */
    updateParticles(deltaTime) {
        const { xMax: extXMax } = this.bounds;
        const speedScale = this.config.speedScale;
        const count = this.config.particleCount;
        
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            
            let x = this.particlePositions[idx + 0];
            let y = this.particlePositions[idx + 1];
            let z = this.particlePositions[idx + 2];
            
            // Get segment at current position (before movement)
            const segBefore = this.getSegmentAtPosition(x, y, z);
            
            // Store original Y and Z for radial calculations (before any updates)
            const yOriginal = y;
            const zOriginal = z;
            
            // Get velocity
            const vel = this.getVelocityAt(x, y);
            
            // Store velocity for renderer (used for droplet stretching, coloring)
            this.particleVelocities[idx + 0] = vel.vx;
            this.particleVelocities[idx + 1] = vel.vy;
            this.particleVelocities[idx + 2] = 0;
            
            // Move particle
            x += vel.vx * speedScale * deltaTime;
            y += vel.vy * speedScale * deltaTime;
            
            // Cylindrical mode: scale radial position based on tube radius change
            // This makes particles expand/contract with the tube geometry
            if (this.config.extrusionMode === 'cylindrical' && segBefore) {
                const segAfter = this.getSegmentAtPosition(x, y, z);
                if (segAfter && segBefore.radius > 0.001 && segAfter.radius > 0.001) {
                    // Check if transitioning between different segments (bifurcation)
                    const centerDiff = Math.abs(segBefore.centerY - segAfter.centerY);
                    const isBifurcation = centerDiff > 0.01;
                    
                    // Calculate radial position relative to ORIGINAL segment's center
                    const dyBefore = yOriginal - segBefore.centerY;
                    const radialDistBefore = Math.sqrt(dyBefore * dyBefore + zOriginal * zOriginal);
                    
                    if (isBifurcation && radialDistBefore > 0.0001) {
                        // Bifurcation: remap from half of inlet to full branch
                        // Upper branch receives particles from top half of inlet (dyBefore > 0)
                        // Lower branch receives particles from bottom half of inlet (dyBefore < 0)
                        
                        const goingToUpperBranch = segAfter.centerY > segBefore.centerY;
                        
                        // Calculate how far the particle is within its "feeding half" of the inlet
                        // dyBefore ranges from 0 to +radius (upper) or 0 to -radius (lower)
                        // Normalize to 0-1 range within that half
                        const halfRadius = segBefore.radius;
                        const distFromCenter = Math.abs(dyBefore);
                        const normalizedInHalf = Math.min(1.0, distFromCenter / halfRadius);
                        
                        // Map to full branch: 0 (center of inlet) -> -radius (bottom of branch)
                        //                     1 (edge of inlet)   -> +radius (top of branch)
                        // So normalized 0-1 maps to -1 to +1 in the branch
                        const mappedDy = (normalizedInHalf * 2 - 1) * segAfter.radius;
                        
                        // Preserve Z proportion
                        const zNormalized = zOriginal / Math.max(0.001, radialDistBefore);
                        const newRadialDist = Math.abs(mappedDy);
                        const newZ = zNormalized * Math.sqrt(Math.max(0, segAfter.radius * segAfter.radius - mappedDy * mappedDy));
                        
                        y = segAfter.centerY + mappedDy;
                        z = zOriginal >= 0 ? Math.abs(newZ) : -Math.abs(newZ);
                        
                    } else if (radialDistBefore > 0.0001) {
                        // Same segment or very small movement - simple radial scaling
                        const normalizedRadial = Math.min(1.0, radialDistBefore / segBefore.radius);
                        const radialDistAfter = normalizedRadial * segAfter.radius;
                        const angle = Math.atan2(zOriginal, dyBefore);
                        
                        y = segAfter.centerY + Math.cos(angle) * radialDistAfter;
                        z = Math.sin(angle) * radialDistAfter;
                    }
                }
            }
            
            // Constrain to tube
            const constrained = this.constrainToTube(x, y, z);
            y = constrained.y;
            z = constrained.z;
            
            // Cylindrical mode: check if particle visual would extend past curved boundary
            // If so, and it's not at the exit, deflect the particle to contour around obstacle
            if (this.config.extrusionMode === 'cylindrical') {
                const { xMax: origXMax } = this.originalBounds;
                const particleRadius = this.config.particleSize || 0.015;
                const lookAheadX = x + particleRadius;
                
                // Only apply contouring if NOT at the mesh exit
                if (lookAheadX < origXMax) {
                    const segAhead = this.getSegmentAtPosition(lookAheadX, y, z);
                    if (segAhead) {
                        const dy = y - segAhead.centerY;
                        const currentRadialDist = Math.sqrt(dy * dy + z * z);
                        const maxAllowedRadius = segAhead.radius - this.config.tubeWallMargin;
                        
                        // If particle visual would extend outside tube at look-ahead position
                        if (currentRadialDist > maxAllowedRadius) {
                            // Deflect: push particle inward radially instead of letting it cross
                            // This simulates fluid contouring around an obstacle
                            const scale = maxAllowedRadius / currentRadialDist;
                            y = segAhead.centerY + dy * scale;
                            z = z * scale;
                        }
                    }
                }
            }
            
            // Update lifetime
            this.particleLifetimes[i] -= deltaTime;
            
            // Check respawn
            const pastOutlet = x > extXMax;
            const expired = this.particleLifetimes[i] <= 0;
            const outsideFlow = !this.isInFlowRegion(x, y, z);
            
            if (pastOutlet || expired || outsideFlow) {
                this.respawnParticle(i, false);
            } else {
                this.particlePositions[idx + 0] = x;
                this.particlePositions[idx + 1] = y;
                // Interpolate Z based on flatten factor (0 = full Z, 1 = Z=0)
                const flattenFactor = this.zFlattenFactor !== undefined ? this.zFlattenFactor : (this.is2DMode ? 1 : 0);
                this.particlePositions[idx + 2] = z * (1 - flattenFactor);
            }
        }
        
        // Update renderer
        if (this.particleRenderer) {
            this.particleRenderer.update(
                this.particlePositions, 
                this.particleVelocities, 
                this.maxSpeed
            );
        }
    }
    
    /**
     * Start animation
     */
    start() {
        if (this.isAnimating) return;
        
        // Initialize particles if needed
        if (!this.particlePositions) {
            this.initializeParticles();
        }
        
        // Create renderer if needed
        if (!this.particleRenderer) {
            this.createParticleRenderer();
        }
        
        this.isAnimating = true;
        this.lastTime = performance.now();
        
        if (this.particleRenderer) {
            this.particleRenderer.setVisible(true);
        }
        
        console.log('Particle animation started');
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            const now = performance.now();
            const deltaTime = Math.min((now - this.lastTime) / 1000, 0.1);
            this.lastTime = now;
            
            this.updateParticles(deltaTime);
            
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
     * Set 2D mode - flattens particles to Z=0 plane
     * @param {boolean} enabled - true for 2D mode, false for 3D mode
     */
    set2DMode(enabled) {
        this.is2DMode = enabled;
        console.log(`ParticleFlow 2D mode: ${enabled ? 'enabled' : 'disabled'}`);
    }
    
    /**
     * Set Z flatten factor for gradual transition
     * @param {number} factor - 0 = full 3D (normal Z), 1 = fully flattened (Z=0)
     */
    setZFlatten(factor) {
        this.zFlattenFactor = Math.max(0, Math.min(1, factor));
    }
    
    /**
     * Toggle visibility
     */
    setVisible(visible) {
        if (this.particleRenderer) {
            this.particleRenderer.setVisible(visible);
        }
        
        if (visible && !this.isAnimating) {
            this.start();
        } else if (!visible && this.isAnimating) {
            this.stop();
        }
    }
    
    /**
     * Set particle color
     */
    setColor(color) {
        this.config.particleColor = color;
        this.config.colorBySpeed = false;
        
        if (this.particleRenderer) {
            this.particleRenderer.setColor(color);
        }
    }
    
    /**
     * Enable/disable speed-based coloring
     */
    setColorBySpeed(enabled) {
        this.config.colorBySpeed = enabled;
        
        if (this.particleRenderer) {
            this.particleRenderer.setColorBySpeed(enabled);
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        // Check if particle type is changing
        if (newConfig.particleType && newConfig.particleType !== this.config.particleType) {
            Object.assign(this.config, newConfig);
            this.setParticleType(newConfig.particleType);
            return;
        }
        
        // Check if particle count is changing
        const countChanging = newConfig.particleCount !== undefined && 
                              newConfig.particleCount !== this.config.particleCount;
        
        Object.assign(this.config, newConfig);
        
        // Update bounds if flow distances changed
        if (newConfig.inflowDistance !== undefined || newConfig.outflowDistance !== undefined) {
            this.bounds.xMin = this.originalBounds.xMin - this.config.inflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin);
            this.bounds.xMax = this.originalBounds.xMax + this.config.outflowDistance * (this.originalBounds.xMax - this.originalBounds.xMin);
        }
        
        // Recreate if count changed
        if (countChanging) {
            const wasAnimating = this.isAnimating;
            if (wasAnimating) this.stop();
            
            this.initializeParticles();
            this.createParticleRenderer();
            
            if (wasAnimating) this.start();
        } else if (this.particleRenderer) {
            // Just update renderer config
            const needsRecreate = this.particleRenderer.updateConfig(newConfig);
            
            if (needsRecreate) {
                const wasAnimating = this.isAnimating;
                if (wasAnimating) this.stop();
                
                this.particleRenderer.create(this.config.particleCount);
                
                if (wasAnimating) this.start();
            }
        }
    }
    
    /**
     * Dispose all resources
     */
    dispose() {
        this.stop();
        
        if (this.particleRenderer) {
            this.particleRenderer.dispose();
            this.particleRenderer = null;
        }
        
        this.meshExtruder.group.remove(this.group);
    }
}
