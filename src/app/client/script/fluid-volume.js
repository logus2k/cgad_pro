import * as THREE from '../library/three.module.min.js';

/**
 * FluidVolume - Renders flowing water inside a tube mesh
 * Single thick mesh where top surface animates with waves, sides connect to it
 */
export class FluidVolume {
    constructor(group, meshExtruder, velocityData, config = {}) {
        this.group = group;
        this.meshExtruder = meshExtruder;
        this.velocityData = velocityData;
        
        this.config = {
            opacity: 0.85,
            flowSpeed: 0.5,
            fillRatio: 0.6,
            waveHeight: 0.06,
            waveSpeed: 2.5,
            // Caribbean sea colors
            deepColor: new THREE.Color(0x006994),
            shallowColor: new THREE.Color(0x40E0D0),
            surfaceColor: new THREE.Color(0x7FDBFF),
            ...config
        };
        
        this.mesh = null;
        this.uniforms = null;
        this.sunLight = null;
    }
    
    /**
     * Create the fluid volume mesh
     */
    create() {
        this.dispose();
        
        const sourceMesh = this.meshExtruder.mesh3D;
        
        if (!sourceMesh || !sourceMesh.geometry) {
            console.warn('FluidVolume: No 3D mesh available');
            return;
        }
        
        const bounds = this.meshExtruder.bounds;
        const origBounds = this.meshExtruder.originalBounds;
        const ySegmentCache = this.meshExtruder.ySegmentCache;
        
        if (!ySegmentCache || ySegmentCache.length === 0) {
            console.warn('FluidVolume: No ySegmentCache');
            return;
        }
        
        // Add sun light for water reflections
        this.createSunLight();
        
        // Setup shared uniforms for animation
        this.uniforms = {
            time: { value: 0 },
            flowSpeed: { value: this.config.flowSpeed },
            waveHeight: { value: this.config.waveHeight },
            waveSpeed: { value: this.config.waveSpeed },
            deepColor: { value: this.config.deepColor },
            shallowColor: { value: this.config.shallowColor },
            surfaceColor: { value: this.config.surfaceColor },
            sunDirection: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
            // Velocity field bounds
            velXMin: { value: origBounds.xMin },
            velXMax: { value: origBounds.xMax },
            velYMin: { value: origBounds.yMin },
            velYMax: { value: origBounds.yMax },
            velocityScale: { value: 1.0 },
            avgSpeed: { value: 1.0 },
            velocityTexture: { value: null },
            hasVelocity: { value: 0.0 }
        };
        
        // Build velocity texture from velocity data
        this.buildVelocityTexture(origBounds);
        
        // Extend bounds for inflow/outflow
        const xRange = origBounds.xMax - origBounds.xMin;
        const extendedXMin = origBounds.xMin - xRange * 0.15;
        const extendedXMax = origBounds.xMax + xRange * 0.25;
        
        // Z depth - small inset from walls
        const zInset = 0.02;
        const yInset = 0.01;
        const zHalf = (bounds.zMax - bounds.zMin) / 2 - zInset;
        
        // Create water geometry and material
        const waterGeometry = this.createWaterGeometry(
            extendedXMin, extendedXMax, 
            origBounds, ySegmentCache, 
            zHalf, this.config.fillRatio,
            yInset
        );
        const waterMaterial = this.createWaterMaterial();
        
        this.mesh = new THREE.Mesh(waterGeometry, waterMaterial);
        
        // Apply same transform as source mesh
        this.mesh.position.copy(sourceMesh.position);
        this.mesh.scale.copy(sourceMesh.scale);
        this.mesh.rotation.copy(sourceMesh.rotation);
        
        this.group.add(this.mesh);
        
        // Keep the original tube mesh visible - water flows inside it
        // (Previously we hid it, but we want to see the Venturi shape)
        if (this.meshExtruder.mesh3D) {
            this.meshExtruder.mesh3D.visible = true;
        }
        
        console.log('FluidVolume: Created unified thick wave mesh');
    }
    
    /**
     * Create sun light for water reflections
     */
    createSunLight() {
        this.sunLight = new THREE.DirectionalLight(0xffffcc, 1.5);
        this.sunLight.position.set(50, 100, 30);
        this.group.add(this.sunLight);
        
        this.ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.group.add(this.ambientLight);
    }
    
    /**
     * Build velocity texture from solver velocity data
     */
    buildVelocityTexture(origBounds) {
        if (!this.velocityData || !this.velocityData.vel) {
            console.log('FluidVolume: No velocity data available');
            return;
        }
        
        // Build a velocity grid like particle-flow does
        const gridResX = 128;
        const gridResY = 64;
        
        const xRange = origBounds.xMax - origBounds.xMin;
        const yRange = origBounds.yMax - origBounds.yMin;
        
        // Initialize grid
        const grid = [];
        for (let i = 0; i < gridResX; i++) {
            grid[i] = [];
            for (let j = 0; j < gridResY; j++) {
                grid[i][j] = { vx: 0, vy: 0, speed: 0, count: 0 };
            }
        }
        
        // Get mesh data - correct structure
        const coords = this.meshExtruder.meshData?.coordinates;
        const connectivity = this.meshExtruder.meshData?.connectivity;
        
        if (!coords || !connectivity) {
            console.log('FluidVolume: No mesh data for velocity grid');
            return;
        }
        
        console.log('FluidVolume: Building grid from', connectivity.length, 'elements');
        
        // Populate grid from element velocities
        for (let e = 0; e < connectivity.length; e++) {
            const elem = connectivity[e];
            if (!elem || elem.length === 0) continue;
            
            // Get element centroid from node coordinates
            let cx = 0, cy = 0;
            let validNodes = 0;
            for (const nodeIdx of elem) {
                if (coords.x[nodeIdx] !== undefined && coords.y[nodeIdx] !== undefined) {
                    cx += coords.x[nodeIdx];
                    cy += coords.y[nodeIdx];
                    validNodes++;
                }
            }
            if (validNodes === 0) continue;
            cx /= validNodes;
            cy /= validNodes;
            
            // Map to grid
            const gi = Math.floor((cx - origBounds.xMin) / xRange * gridResX);
            const gj = Math.floor((cy - origBounds.yMin) / yRange * gridResY);
            
            if (gi >= 0 && gi < gridResX && gj >= 0 && gj < gridResY) {
                const vel = this.velocityData.vel[e];
                if (vel) {
                    const cell = grid[gi][gj];
                    cell.vx += vel[0] || 0;
                    cell.vy += vel[1] || 0;
                    cell.speed += this.velocityData.abs_vel ? this.velocityData.abs_vel[e] : Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
                    cell.count++;
                }
            }
        }
        
        // Average and find max
        let maxSpeed = 0;
        for (let i = 0; i < gridResX; i++) {
            for (let j = 0; j < gridResY; j++) {
                const cell = grid[i][j];
                if (cell.count > 0) {
                    cell.vx /= cell.count;
                    cell.vy /= cell.count;
                    cell.speed /= cell.count;
                    maxSpeed = Math.max(maxSpeed, cell.speed);
                }
            }
        }
        
        console.log('FluidVolume: Grid built, maxSpeed =', maxSpeed.toFixed(3));
        
        // Create texture data with cumulative flow distance
        const data = new Float32Array(gridResX * gridResY * 4);
        
        // First, compute average speed at each X position (for cumulative integration)
        const avgSpeedAtX = [];
        for (let i = 0; i < gridResX; i++) {
            let totalSpeed = 0;
            let count = 0;
            for (let j = 0; j < gridResY; j++) {
                if (grid[i][j].count > 0) {
                    totalSpeed += grid[i][j].speed;
                    count++;
                }
            }
            avgSpeedAtX[i] = count > 0 ? totalSpeed / count : 1.0;
            // Fill gaps with neighbor values
            if (avgSpeedAtX[i] === 0 && i > 0) {
                avgSpeedAtX[i] = avgSpeedAtX[i-1];
            }
        }
        
        // Compute travel time to reach each X position from inlet
        // travelTime[i] = integral of (1/velocity) from 0 to i
        // This represents the actual time a water parcel takes to reach position i
        const travelTime = [0];
        const dx = xRange / gridResX;
        for (let i = 1; i < gridResX; i++) {
            const speed = avgSpeedAtX[i];
            // Time to cross this cell = distance / speed
            const dt = speed > 0.1 ? dx / speed : dx / 0.1;
            travelTime[i] = travelTime[i - 1] + dt;
        }
        
        const maxTravelTime = travelTime[gridResX - 1];
        console.log('FluidVolume: Travel time from inlet to outlet:', maxTravelTime.toFixed(3), 'seconds');
        
        // DON'T normalize - keep actual time values for proper animation sync
        // Store raw travel time, we'll use it directly in shader

        for (let j = 0; j < gridResY; j++) {
            for (let i = 0; i < gridResX; i++) {
                const cell = grid[i][j];
                const idx = (j * gridResX + i) * 4;
                
                // Normalize speed for display purposes
                const normSpeed = maxSpeed > 0 ? cell.speed / maxSpeed : 0;
                
                // Store actual travel time (not normalized) for proper sync
                data[idx] = travelTime[i];        // R = actual travel time in seconds
                data[idx + 1] = normSpeed;        // G = normalized speed for visual effects
                data[idx + 2] = maxTravelTime;    // B = max travel time (same for all, for reference)
                data[idx + 3] = avgSpeedAtX[i];   // A = actual speed at this X
            }
        }
        
        // Create texture
        const texture = new THREE.DataTexture(data, gridResX, gridResY, THREE.RGBAFormat, THREE.FloatType);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        
        this.uniforms.velocityTexture.value = texture;
        this.uniforms.hasVelocity.value = 1.0;
        this.uniforms.velocityScale.value = maxSpeed;
        
        // Compute average speed for uniform flow
        let totalSpeed = 0;
        let count = 0;
        for (let i = 0; i < gridResX; i++) {
            if (avgSpeedAtX[i] > 0) {
                totalSpeed += avgSpeedAtX[i];
                count++;
            }
        }
        const avgSpeed = count > 0 ? totalSpeed / count : maxSpeed;
        this.uniforms.avgSpeed = { value: avgSpeed };
        console.log('FluidVolume: Average speed =', avgSpeed.toFixed(3));
        
        // Debug: sample velocity at different X positions
        console.log('FluidVolume velocity samples:');
        const midJ = Math.floor(gridResY / 2);
        for (const t of [0.1, 0.3, 0.5, 0.7, 0.9]) {
            const i = Math.floor(t * (gridResX - 1));
            const cell = grid[i][midJ];
            console.log(`  X=${t.toFixed(1)}: speed=${cell.speed.toFixed(3)}`);
        }
        
        console.log('FluidVolume: Built velocity texture, maxSpeed =', maxSpeed.toFixed(3));
    }
    
    /**
     * Fallback: build velocity texture using just vel array (without mesh geometry)
     * Assumes velocities are ordered by element index which roughly follows X position
     */
    buildVelocityTextureFromVelArray(origBounds, gridResX, gridResY) {
        const vel = this.velocityData.vel;
        const absVel = this.velocityData.abs_vel;
        
        // Find max speed
        let maxSpeed = 0;
        for (let i = 0; i < absVel.length; i++) {
            if (absVel[i] > maxSpeed) maxSpeed = absVel[i];
        }
        
        console.log('FluidVolume: Fallback method, maxSpeed =', maxSpeed.toFixed(3), 'numElements =', vel.length);
        
        // Create texture - for fallback, we'll create a gradient based on typical Venturi flow
        // Faster in middle (throat), slower at ends
        const data = new Float32Array(gridResX * gridResY * 4);
        
        for (let j = 0; j < gridResY; j++) {
            for (let i = 0; i < gridResX; i++) {
                const idx = (j * gridResX + i) * 4;
                const tx = i / (gridResX - 1); // 0 to 1 along X
                
                // Sample from velocity array - map texture X to element index
                const elemIdx = Math.floor(tx * (vel.length - 1));
                const speed = absVel[elemIdx] || 0;
                const normSpeed = maxSpeed > 0 ? speed / maxSpeed : 0.5;
                
                data[idx] = vel[elemIdx]?.[0] || 0;
                data[idx + 1] = vel[elemIdx]?.[1] || 0;
                data[idx + 2] = normSpeed;
                data[idx + 3] = 1.0;
            }
        }
        
        // Create texture
        const texture = new THREE.DataTexture(data, gridResX, gridResY, THREE.RGBAFormat, THREE.FloatType);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        
        this.uniforms.velocityTexture.value = texture;
        this.uniforms.hasVelocity.value = 1.0;
        this.uniforms.velocityScale.value = maxSpeed;
        
        // Debug samples
        console.log('FluidVolume velocity samples (fallback):');
        for (const t of [0.1, 0.3, 0.5, 0.7, 0.9]) {
            const elemIdx = Math.floor(t * (vel.length - 1));
            const speed = absVel[elemIdx] || 0;
            console.log(`  X=${t.toFixed(1)}: speed=${speed.toFixed(3)}, normalized=${(maxSpeed > 0 ? speed/maxSpeed : 0).toFixed(3)}`);
        }
    }

    /**
     * Get Y bounds at X position from segment cache
     */
    getSegmentAtX(x, origBounds, ySegmentCache) {
        const t = (x - origBounds.xMin) / (origBounds.xMax - origBounds.xMin);
        const clampedT = Math.max(0, Math.min(1, t));
        
        const index = Math.floor(clampedT * (ySegmentCache.length - 1));
        const safeIndex = Math.max(0, Math.min(ySegmentCache.length - 1, index));
        
        const segArray = ySegmentCache[safeIndex];
        const seg = Array.isArray(segArray) ? segArray[0] : segArray;
        
        return {
            yMin: seg?.yMin ?? origBounds.yMin,
            yMax: seg?.yMax ?? origBounds.yMax,
            centerY: seg?.centerY ?? (origBounds.yMin + origBounds.yMax) / 2,
            radius: seg?.radius ?? (origBounds.yMax - origBounds.yMin) / 2
        };
    }
    
    /**
     * Create unified water geometry - single thick mesh
     * Top surface and top edges of sides animate with waves
     * Bottom and lower portions are fixed
     */
    createWaterGeometry(xMin, xMax, origBounds, ySegmentCache, zHalf, fillRatio, yInset = 0.01) {
        const numXSlices = 80;
        const numZSlices = 20;
        
        const vertices = [];
        const indices = [];
        const uvs = [];
        const depths = [];
        const isSurface = [];
        const baseYs = [];
        const wavePhaseZs = []; // Z coordinate to use for wave phase calculation
        
        let idx = 0;
        
        // ============ TOP SURFACE (animated waves) ============
        const topStart = idx;
        for (let i = 0; i <= numXSlices; i++) {
            const tx = i / numXSlices;
            const x = xMin + tx * (xMax - xMin);
            const seg = this.getSegmentAtX(x, origBounds, ySegmentCache);
            const waterY = seg.yMin + (seg.yMax - seg.yMin) * fillRatio;
            
            for (let j = 0; j <= numZSlices; j++) {
                const tz = j / numZSlices;
                const z = -zHalf + tz * zHalf * 2;
                vertices.push(x, waterY, z);
                uvs.push(tx, tz);
                depths.push(1.0);
                isSurface.push(1.0);
                baseYs.push(waterY);
                wavePhaseZs.push(z); // Use actual Z
                idx++;
            }
        }
        
        // Top surface faces
        for (let i = 0; i < numXSlices; i++) {
            for (let j = 0; j < numZSlices; j++) {
                const a = topStart + i * (numZSlices + 1) + j;
                const b = a + 1;
                const c = topStart + (i + 1) * (numZSlices + 1) + j;
                const d = c + 1;
                indices.push(a, c, d, a, d, b);
            }
        }
        
        // ============ BOTTOM SURFACE (fixed) ============
        const bottomStart = idx;
        for (let i = 0; i <= numXSlices; i++) {
            const tx = i / numXSlices;
            const x = xMin + tx * (xMax - xMin);
            const seg = this.getSegmentAtX(x, origBounds, ySegmentCache);
            const yBottom = seg.yMin + yInset;
            
            for (let j = 0; j <= numZSlices; j++) {
                const tz = j / numZSlices;
                const z = -zHalf + tz * zHalf * 2;
                vertices.push(x, yBottom, z);
                uvs.push(tx, tz);
                depths.push(0.0);
                isSurface.push(0.0);
                baseYs.push(yBottom);
                wavePhaseZs.push(z);
                idx++;
            }
        }
        
        // Bottom faces
        for (let i = 0; i < numXSlices; i++) {
            for (let j = 0; j < numZSlices; j++) {
                const a = bottomStart + i * (numZSlices + 1) + j;
                const b = a + 1;
                const c = bottomStart + (i + 1) * (numZSlices + 1) + j;
                const d = c + 1;
                indices.push(a, d, c, a, b, d);
            }
        }
        
        // ============ FRONT SIDE (z = -zHalf) ============
        // Bottom edge fixed, top edge animates with waves
        const frontStart = idx;
        for (let i = 0; i <= numXSlices; i++) {
            const tx = i / numXSlices;
            const x = xMin + tx * (xMax - xMin);
            const seg = this.getSegmentAtX(x, origBounds, ySegmentCache);
            const yBot = seg.yMin + yInset;
            const yTop = seg.yMin + (seg.yMax - seg.yMin) * fillRatio;
            
            // Bottom vertex (fixed)
            vertices.push(x, yBot, -zHalf);
            uvs.push(tx, 0.0);
            depths.push(0.0);
            isSurface.push(0.0);
            baseYs.push(yBot);
            wavePhaseZs.push(-zHalf);
            idx++;
            
            // Top vertex (animated with waves) - use -zHalf for wave phase
            vertices.push(x, yTop, -zHalf);
            uvs.push(tx, 1.0);
            depths.push(1.0);
            isSurface.push(1.0);
            baseYs.push(yTop);
            wavePhaseZs.push(-zHalf); // Match front edge of top surface
            idx++;
        }
        
        // Front faces
        for (let i = 0; i < numXSlices; i++) {
            const a = frontStart + i * 2;
            const b = frontStart + i * 2 + 1;
            const c = frontStart + (i + 1) * 2;
            const d = frontStart + (i + 1) * 2 + 1;
            indices.push(a, b, d, a, d, c);
        }
        
        // ============ BACK SIDE (z = +zHalf) ============
        const backStart = idx;
        for (let i = 0; i <= numXSlices; i++) {
            const tx = i / numXSlices;
            const x = xMin + tx * (xMax - xMin);
            const seg = this.getSegmentAtX(x, origBounds, ySegmentCache);
            const yBot = seg.yMin + yInset;
            const yTop = seg.yMin + (seg.yMax - seg.yMin) * fillRatio;
            
            // Bottom vertex (fixed)
            vertices.push(x, yBot, zHalf);
            uvs.push(tx, 0.0);
            depths.push(0.0);
            isSurface.push(0.0);
            baseYs.push(yBot);
            wavePhaseZs.push(zHalf);
            idx++;
            
            // Top vertex (animated) - use +zHalf for wave phase
            vertices.push(x, yTop, zHalf);
            uvs.push(tx, 1.0);
            depths.push(1.0);
            isSurface.push(1.0);
            baseYs.push(yTop);
            wavePhaseZs.push(zHalf); // Match back edge of top surface
            idx++;
        }
        
        // Back faces
        for (let i = 0; i < numXSlices; i++) {
            const a = backStart + i * 2;
            const b = backStart + i * 2 + 1;
            const c = backStart + (i + 1) * 2;
            const d = backStart + (i + 1) * 2 + 1;
            indices.push(a, c, d, a, d, b);
        }
        
        // ============ LEFT CAP (x = xMin) ============
        const leftStart = idx;
        const segL = this.getSegmentAtX(xMin, origBounds, ySegmentCache);
        const yBotL = segL.yMin + yInset;
        const yTopL = segL.yMin + (segL.yMax - segL.yMin) * fillRatio;
        
        for (let j = 0; j <= numZSlices; j++) {
            const tz = j / numZSlices;
            const z = -zHalf + tz * zHalf * 2;
            
            // Bottom vertex
            vertices.push(xMin, yBotL, z);
            uvs.push(tz, 0.0);
            depths.push(0.0);
            isSurface.push(0.0);
            baseYs.push(yBotL);
            wavePhaseZs.push(z);
            idx++;
            
            // Top vertex (animated) - use actual Z for wave phase
            vertices.push(xMin, yTopL, z);
            uvs.push(tz, 1.0);
            depths.push(1.0);
            isSurface.push(1.0);
            baseYs.push(yTopL);
            wavePhaseZs.push(z); // Match left edge of top surface at this Z
            idx++;
        }
        
        // Left cap faces
        for (let j = 0; j < numZSlices; j++) {
            const a = leftStart + j * 2;
            const b = leftStart + j * 2 + 1;
            const c = leftStart + (j + 1) * 2;
            const d = leftStart + (j + 1) * 2 + 1;
            indices.push(a, b, d, a, d, c);
        }
        
        // ============ RIGHT CAP (x = xMax) ============
        const rightStart = idx;
        const segR = this.getSegmentAtX(xMax, origBounds, ySegmentCache);
        const yBotR = segR.yMin + yInset;
        const yTopR = segR.yMin + (segR.yMax - segR.yMin) * fillRatio;
        
        for (let j = 0; j <= numZSlices; j++) {
            const tz = j / numZSlices;
            const z = -zHalf + tz * zHalf * 2;
            
            // Bottom vertex
            vertices.push(xMax, yBotR, z);
            uvs.push(tz, 0.0);
            depths.push(0.0);
            isSurface.push(0.0);
            baseYs.push(yBotR);
            wavePhaseZs.push(z);
            idx++;
            
            // Top vertex (animated) - use actual Z for wave phase
            vertices.push(xMax, yTopR, z);
            uvs.push(tz, 1.0);
            depths.push(1.0);
            isSurface.push(1.0);
            baseYs.push(yTopR);
            wavePhaseZs.push(z); // Match right edge of top surface at this Z
            idx++;
        }
        
        // Right cap faces
        for (let j = 0; j < numZSlices; j++) {
            const a = rightStart + j * 2;
            const b = rightStart + j * 2 + 1;
            const c = rightStart + (j + 1) * 2;
            const d = rightStart + (j + 1) * 2 + 1;
            indices.push(a, c, d, a, d, b);
        }
        
        // Build geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        geometry.setAttribute('depth', new THREE.Float32BufferAttribute(depths, 1));
        geometry.setAttribute('isSurface', new THREE.Float32BufferAttribute(isSurface, 1));
        geometry.setAttribute('baseY', new THREE.Float32BufferAttribute(baseYs, 1));
        geometry.setAttribute('wavePhaseZ', new THREE.Float32BufferAttribute(wavePhaseZs, 1));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        // Debug: count surface vs non-surface vertices
        let surfaceCount = 0;
        let nonSurfaceCount = 0;
        for (let i = 0; i < isSurface.length; i++) {
            if (isSurface[i] > 0.5) surfaceCount++;
            else nonSurfaceCount++;
        }
        console.log('FluidVolume geometry debug:');
        console.log('  Total vertices:', vertices.length / 3);
        console.log('  Surface vertices (isSurface=1):', surfaceCount);
        console.log('  Non-surface vertices (isSurface=0):', nonSurfaceCount);
        console.log('  Top surface vertices:', (numXSlices + 1) * (numZSlices + 1));
        console.log('  Front/back side TOP vertices:', (numXSlices + 1) * 2);
        console.log('  Left/right cap TOP vertices:', (numZSlices + 1) * 2);
        
        return geometry;
    }
    
    /**
     * Create unified water material with velocity-based flow and surface waves
     */
    createWaterMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: `
                uniform float time;
                uniform float velXMin;
                uniform float velXMax;
                uniform sampler2D velocityTexture;
                uniform float hasVelocity;
                uniform float velocityScale;
                uniform float avgSpeed;
                uniform float flowSpeed;
                uniform float waveHeight;
                
                attribute float depth;
                attribute float isSurface;
                attribute float baseY;
                
                varying vec3 vWorldPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                varying float vDepth;
                varying float vFlowDistance;
                varying float vLocalSpeed;
                
                void main() {
                    vUv = uv;
                    vDepth = depth;
                    
                    vec3 pos = position;
                    vec3 norm = normal;
                    
                    // Sample velocity texture at this X position
                    vFlowDistance = 0.0;
                    vLocalSpeed = 0.5;
                    if (hasVelocity > 0.5) {
                        float uvX = clamp((pos.x - velXMin) / (velXMax - velXMin), 0.0, 1.0);
                        vec4 velSample = texture2D(velocityTexture, vec2(uvX, 0.5));
                        vFlowDistance = velSample.r;
                        vLocalSpeed = velSample.g;
                    }
                    
                    // Apply waves to top surface
                    if (isSurface > 0.5) {
                        float flowOffset = time * avgSpeed * flowSpeed;
                        float wavePhase = pos.x * 3.0 - flowOffset;
                        
                        // Waves
                        float wave1 = sin(wavePhase * 0.5) * 0.4;
                        float wave2 = sin(wavePhase * 0.8 + 1.0) * 0.3;
                        float wave3 = sin(wavePhase * 1.2 + 2.0) * 0.2;
                        float wave4 = sin(wavePhase * 1.8) * 0.1;
                        
                        float totalWave = (wave1 + wave2 + wave3 + wave4) * waveHeight;
                        pos.y = baseY + totalWave;
                        
                        // Wave normal
                        float dx = cos(wavePhase * 0.5) * 0.5 * 0.4 +
                                   cos(wavePhase * 0.8 + 1.0) * 0.8 * 0.3 +
                                   cos(wavePhase * 1.2 + 2.0) * 1.2 * 0.2 +
                                   cos(wavePhase * 1.8) * 1.8 * 0.1;
                        dx *= waveHeight * 3.0;
                        norm = normalize(vec3(-dx, 1.0, 0.0));
                    }
                    
                    vNormal = normalize(normalMatrix * norm);
                    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
                    vWorldPosition = worldPos.xyz;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float flowSpeed;
                uniform float velocityScale;
                uniform float avgSpeed;
                uniform vec3 deepColor;
                uniform vec3 shallowColor;
                uniform vec3 sunDirection;
                
                varying vec3 vWorldPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                varying float vDepth;
                varying float vFlowDistance;
                varying float vLocalSpeed;
                
                float hash(vec3 p) {
                    p = fract(p * 0.3183099 + 0.1);
                    p *= 17.0;
                    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
                }
                
                float noise(vec3 p) {
                    vec3 i = floor(p);
                    vec3 f = fract(p);
                    f = f * f * (3.0 - 2.0 * f);
                    return mix(
                        mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
                            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
                        f.z
                    );
                }
                
                void main() {
                    // Use average speed from solver, scaled by flowSpeed (matches particle speedScale)
                    float flowOffset = time * avgSpeed * flowSpeed;
                    
                    vec3 flowPos = vWorldPosition * 3.0;
                    flowPos.x -= flowOffset;
                    
                    // Base color gradient
                    vec3 color = mix(deepColor, shallowColor, vDepth);
                    
                    // Flowing noise pattern
                    float n1 = noise(flowPos);
                    float n2 = noise(flowPos * 2.0 + vec3(100.0, 50.0, 25.0));
                    float flowNoise = n1 * 0.5 + n2 * 0.3;
                    color = mix(color, shallowColor, flowNoise * 0.25);
                    
                    // Caustics
                    vec3 causticPos = flowPos * 1.5;
                    float c1 = noise(causticPos + vec3(0.0, time * 0.3, 0.0));
                    float c2 = noise(causticPos * 1.3 + vec3(0.0, 0.0, time * 0.2));
                    float caustic = c1 * c2;
                    caustic = smoothstep(0.2, 0.5, caustic);
                    color += shallowColor * caustic * (0.3 + vLocalSpeed * 0.4);
                    
                    // Surface effects on top
                    if (vDepth > 0.9) {
                        vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                        float fresnel = pow(1.0 - max(0.0, dot(viewDir, vNormal)), 3.0);
                        color = mix(color, shallowColor * 1.3, fresnel * 0.15);
                        
                        vec3 reflectDir = reflect(-sunDirection, vNormal);
                        float spec = pow(max(0.0, dot(viewDir, reflectDir)), 60.0);
                        color += vec3(1.0, 1.0, 0.95) * spec * 0.3;
                    }
                    
                    gl_FragColor = vec4(color, 0.95);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: true
        });
    }
    
    /**
     * Update animation
     */
    update(deltaTime) {
        if (this.uniforms) {
            this.uniforms.time.value += deltaTime || 0.016;
        }
    }
    
    /**
     * Set visibility
     */
    setVisible(visible) {
        if (this.mesh) this.mesh.visible = visible;
    }
    
    /**
     * Dispose resources
     */
    dispose() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            if (this.mesh.material) this.mesh.material.dispose();
            this.mesh = null;
        }
        if (this.sunLight) {
            this.group.remove(this.sunLight);
            this.sunLight = null;
        }
        if (this.ambientLight) {
            this.group.remove(this.ambientLight);
            this.ambientLight = null;
        }
        this.uniforms = null;
    }
}
