import * as THREE from '../library/three.module.min.js';

/**
 * MeshExtruderRect - Rectangular extrusion for 2D FEM meshes
 * 
 * Creates a hollow 3D tube by extruding only the BOUNDARY edges of the 2D mesh.
 * Inlet (xMin) and outlets (xMax) are left open for particle flow.
 * 
 * Supports incremental color updates during solve.
 */
export class MeshExtruderRect {
    constructor(scene, meshData, solutionData = null, config = {}) {
        this.scene = scene;
        this.meshData = meshData;
        this.solutionData = solutionData;
        
        this.config = {
            show2DMesh: true,
            show3DExtrusion: false,
            zFactor: 5.0,
            extrusionOpacity: 0.8,
            ...config
        };
        
        this.mesh2D = null;
        this.mesh3D = null;
        this.group = new THREE.Group();
        
        this.geometryCreated = false;
        
        // Boundary edge data (computed once, used for geometry and color updates)
        this.boundaryEdges = null;
        this.wallNodeMap = null;  // Maps vertex index -> mesh node ID
        
        // Calculate bounds
        this.bounds = this.calculateBounds();
        
        this.originalBounds = {
            xMin: this.bounds.xMin,
            xMax: this.bounds.xMax,
            yMin: this.bounds.yMin,
            yMax: this.bounds.yMax
        };
        
        this.scene.add(this.group);
        
        console.log('MeshExtruderRect initialized');
    }
    
    /**
     * Calculate mesh bounds and Z extent
     */
    calculateBounds() {
        const coords = this.meshData.coordinates;
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        for (let i = 0; i < coords.x.length; i++) {
            if (coords.x[i] < xMin) xMin = coords.x[i];
            if (coords.x[i] > xMax) xMax = coords.x[i];
            if (coords.y[i] < yMin) yMin = coords.y[i];
            if (coords.y[i] > yMax) yMax = coords.y[i];
        }
        
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const maxDim = Math.max(xRange, yRange);
        const zExtent = maxDim * this.config.zFactor;
        
        console.log(`   Bounds: X[${xMin.toFixed(3)}, ${xMax.toFixed(3)}], Y[${yMin.toFixed(3)}, ${yMax.toFixed(3)}]`);
        console.log(`   Z extent: +/-${(zExtent/2).toFixed(3)} (factor: ${this.config.zFactor}x)`);
        
        return { 
            xMin, xMax, yMin, yMax, 
            zMin: -zExtent / 2, 
            zMax: zExtent / 2,
            maxDim
        };
    }
    
    /**
     * Build segment cache for rectangular SDF (same approach as cylindrical)
     * For each X position, stores multiple Y segments (to handle bifurcation)
     */
    buildYRangeCache() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        const { xMin, xMax } = this.originalBounds;
        
        this.cacheResolution = 500;
        this.ySegmentCache = new Array(this.cacheResolution);
        
        const xRange = xMax - xMin;
        const gapThreshold = 0.05;  // Minimum gap to consider separate segments
        
        console.log('   Building Y-segment cache for rectangular SDF...');
        
        for (let i = 0; i < this.cacheResolution; i++) {
            const x = xMin + (i / (this.cacheResolution - 1)) * xRange;
            this.ySegmentCache[i] = this.computeYSegmentsAtX(x, coords, conn, gapThreshold);
        }
        
        // Log segment info at a few X positions
        const midIdx = Math.floor(this.cacheResolution / 2);
        const endIdx = this.cacheResolution - 10;
        console.log(`   At inlet: ${this.ySegmentCache[10]?.length || 0} segments`);
        console.log(`   At middle: ${this.ySegmentCache[midIdx]?.length || 0} segments`);
        console.log(`   At outlet: ${this.ySegmentCache[endIdx]?.length || 0} segments`);
    }
    
    /**
     * Compute Y segments at a given X position (same logic as cylindrical SDF)
     */
    computeYSegmentsAtX(x, coords, conn, gapThreshold) {
        const yValues = [];
        const tolerance = 0.001;
        
        // Find all elements that contain this X
        for (const elem of conn) {
            let elemXMin = Infinity, elemXMax = -Infinity;
            let elemYMin = Infinity, elemYMax = -Infinity;
            
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                const nx = coords.x[nodeId];
                const ny = coords.y[nodeId];
                
                if (nx < elemXMin) elemXMin = nx;
                if (nx > elemXMax) elemXMax = nx;
                if (ny < elemYMin) elemYMin = ny;
                if (ny > elemYMax) elemYMax = ny;
            }
            
            if (x >= elemXMin - tolerance && x <= elemXMax + tolerance) {
                yValues.push({ min: elemYMin, max: elemYMax });
            }
        }
        
        if (yValues.length === 0) return null;
        
        // Sort by Y min
        yValues.sort((a, b) => a.min - b.min);
        
        // Merge overlapping ranges, but keep separate segments if gap > threshold
        const segments = [];
        let current = { ...yValues[0] };
        
        for (let i = 1; i < yValues.length; i++) {
            const next = yValues[i];
            if (next.min <= current.max + gapThreshold) {
                // Overlapping or close - merge
                current.max = Math.max(current.max, next.max);
            } else {
                // Gap detected - save current and start new segment
                segments.push({ yMin: current.min, yMax: current.max });
                current = { ...next };
            }
        }
        segments.push({ yMin: current.min, yMax: current.max });
        
        return segments;
    }
    
    /**
     * Get Y segments at a given X position (from cache)
     */
    getYSegmentsAtX(x) {
        const { xMin, xMax } = this.originalBounds;
        
        if (x < xMin || x > xMax) return null;
        
        const t = (x - xMin) / (xMax - xMin);
        const idx = Math.floor(t * (this.cacheResolution - 1));
        
        if (idx < 0 || idx >= this.cacheResolution) return null;
        
        return this.ySegmentCache[idx];
    }
    
    /**
     * Rectangular SDF with multiple Y segments (handles bifurcation)
     * Returns distance to nearest wall, negative inside, positive outside
     */
    rectangularSDF(x, y, z) {
        const { xMin, xMax } = this.originalBounds;
        const { zMin, zMax } = this.bounds;
        
        // Outside X range
        if (x < xMin || x > xMax) {
            return 1.0;
        }
        
        const segments = this.getYSegmentsAtX(x);
        if (!segments || segments.length === 0) {
            return 1.0;
        }
        
        // Find the segment that contains this Y, or the closest one
        let minDist = Infinity;
        
        for (const seg of segments) {
            const { yMin, yMax } = seg;
            
            // Distance to Y bounds of this segment
            const dTop = y - yMax;      // positive when above
            const dBottom = yMin - y;   // positive when below
            const dY = Math.max(dTop, dBottom);  // positive when outside Y range
            
            // Distance to Z bounds
            const dFront = z - zMax;
            const dBack = zMin - z;
            const dZ = Math.max(dFront, dBack);  // positive when outside Z range
            
            // If inside both Y and Z bounds, return max (closest wall, negative)
            // If outside, return positive distance
            const dist = Math.max(dY, dZ);
            
            if (dist < minDist) {
                minDist = dist;
            }
        }
        
        return minDist;
    }
    
    // =========================================================================
    // Geometry Creation
    // =========================================================================
    
    createGeometryOnly() {
        this.buildYRangeCache();
        this.create2DGeometry();
        this.create3DGeometry();
        this.geometryCreated = true;
        console.log('Geometry created (awaiting solution for colors)');
    }
    
    create2DGeometry() {
        if (this.mesh2D) {
            this.group.remove(this.mesh2D);
            this.mesh2D.geometry.dispose();
            this.mesh2D.material.dispose();
        }
        
        const geometry = this.createQuad8Geometry2D();
        this.applyNeutralColors(geometry);
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
        });
        
        this.mesh2D = new THREE.Mesh(geometry, material);
        this.mesh2D.visible = this.config.show2DMesh;
        
        this.fitMeshToView(this.mesh2D);
        this.group.add(this.mesh2D);
        
        console.log('2D geometry created');
    }
    
    async create3DGeometry() {
        if (this.mesh3D) {
            this.group.remove(this.mesh3D);
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
        
        console.log('Creating 3D rectangular tube via Marching Cubes...');
        
        const iso = await this.loadIsosurface();
        
        const { xMin, xMax, yMin, yMax, zMin, zMax } = this.bounds;
        
        // Add margin for marching cubes
        const margin = 0.05;
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const zRange = zMax - zMin;
        
        const mcBounds = {
            xMin: xMin - xRange * margin,
            xMax: xMax + xRange * margin,
            yMin: yMin - yRange * margin,
            yMax: yMax + yRange * margin,
            zMin: zMin - zRange * margin,
            zMax: zMax + zRange * margin
        };
        
        const mcXRange = mcBounds.xMax - mcBounds.xMin;
        const mcYRange = mcBounds.yMax - mcBounds.yMin;
        const mcZRange = mcBounds.zMax - mcBounds.zMin;
        
        // Resolution for marching cubes
        const resX = 120;
        const resY = 80;
        const resZ = 80;
        
        const sdfFunc = (gx, gy, gz) => {
            const x = mcBounds.xMin + (gx / resX) * mcXRange;
            const y = mcBounds.yMin + (gy / resY) * mcYRange;
            const z = mcBounds.zMin + (gz / resZ) * mcZRange;
            return this.rectangularSDF(x, y, z);
        };
        
        const result = iso.surfaceNets(
            [resX + 1, resY + 1, resZ + 1],
            sdfFunc,
            [[0, 0, 0], [resX, resY, resZ]]
        );
        
        if (!result.positions || result.positions.length === 0) {
            console.warn('Marching cubes produced no geometry');
            return;
        }
        
        const geometry = new THREE.BufferGeometry();
        
        // Convert positions from grid coords to world coords
        const positions = new Float32Array(result.positions.length * 3);
        for (let i = 0; i < result.positions.length; i++) {
            const [gx, gy, gz] = result.positions[i];
            positions[i * 3 + 0] = mcBounds.xMin + (gx / resX) * mcXRange;
            positions[i * 3 + 1] = mcBounds.yMin + (gy / resY) * mcYRange;
            positions[i * 3 + 2] = mcBounds.zMin + (gz / resZ) * mcZRange;
        }
        
        // Convert indices
        let indices = new Uint32Array(result.cells.length * 3);
        for (let i = 0; i < result.cells.length; i++) {
            indices[i * 3 + 0] = result.cells[i][0];
            indices[i * 3 + 1] = result.cells[i][1];
            indices[i * 3 + 2] = result.cells[i][2];
        }
        
        // Remove end caps (inlet and outlet)
        indices = this.removeEndCaps(positions, indices);
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        // Apply colors based on X position (gradient like SDF does)
        this.addGradientColors(geometry);
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: this.config.extrusionOpacity
        });
        
        this.mesh3D = new THREE.Mesh(geometry, material);
        this.mesh3D.visible = this.config.show3DExtrusion;
        
        this.fitMeshToView(this.mesh3D);
        this.group.add(this.mesh3D);
        
        console.log(`   Created ${result.positions.length} vertices, ${indices.length / 3} triangles`);
        console.log('3D rectangular tube created');
    }
    
    /**
     * Remove triangles at inlet (xMin) and outlet (xMax)
     */
    removeEndCaps(positions, indices) {
        const { xMin, xMax } = this.originalBounds;
        const tolerance = (xMax - xMin) * 0.02;
        
        const newIndices = [];
        let removedInlet = 0, removedOutlet = 0;
        
        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i + 0];
            const i1 = indices[i + 1];
            const i2 = indices[i + 2];
            
            const x0 = positions[i0 * 3 + 0];
            const x1 = positions[i1 * 3 + 0];
            const x2 = positions[i2 * 3 + 0];
            
            const allAtXMin = x0 < xMin + tolerance && x1 < xMin + tolerance && x2 < xMin + tolerance;
            const allAtXMax = x0 > xMax - tolerance && x1 > xMax - tolerance && x2 > xMax - tolerance;
            
            if (allAtXMin) {
                removedInlet++;
            } else if (allAtXMax) {
                removedOutlet++;
            } else {
                newIndices.push(i0, i1, i2);
            }
        }
        
        console.log(`   Removed end caps: inlet=${removedInlet}, outlet=${removedOutlet} triangles`);
        
        return new Uint32Array(newIndices);
    }
    
    /**
     * Load isosurface library (same as SDF version)
     */
    async loadIsosurface() {
        if (window.isosurface) {
            return window.isosurface;
        }
        
        const module = await import('https://esm.sh/isosurface@1.0.0');
        window.isosurface = module;
        return module;
    }
    
    /**
     * Add gradient colors based on X position
     */
    addGradientColors(geometry) {
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        const { xMin, xMax } = this.originalBounds;
        const xRange = xMax - xMin;
        
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const t = (x - xMin) / xRange;
            const color = this.valueToColor(1 - t);  // Red at inlet, blue at outlet
            
            colors[i * 3 + 0] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    applyNeutralColors(geometry) {
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        for (let i = 0; i < positions.count; i++) {
            colors[i * 3 + 0] = 0.75;
            colors[i * 3 + 1] = 0.75;
            colors[i * 3 + 2] = 0.75;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    // =========================================================================
    // Color Updates (O(n) using wallNodeMap)
    // =========================================================================
    
    updateSolutionColors(solutionData) {
        this.solutionData = solutionData;
        
        if (this.mesh2D) {
            this.updateMesh2DColors();
        }
        
        if (this.mesh3D) {
            this.updateMesh3DColors();
        }
    }
    
    updateSolutionIncremental(updateData) {
        const { solution_values, chunk_info } = updateData;
        
        this.solutionData = {
            values: solution_values,
            range: [chunk_info.min, chunk_info.max]
        };
        
        if (this.mesh2D) {
            this.updateMesh2DColors();
        }
        
        if (this.mesh3D) {
            this.updateMesh3DColors();
        }
    }
    
    updateMesh2DColors() {
        if (!this.mesh2D || !this.solutionData) return;
        
        const geometry = this.mesh2D.geometry;
        const colors = geometry.attributes.color.array;
        
        const values = this.solutionData.values;
        const [min, max] = this.solutionData.range;
        const conn = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (const elem of conn) {
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                const value = values[nodeId] || 0;
                const normalized = (max > min) ? (value - min) / (max - min) : 0.5;
                const color = this.valueToColor(normalized);
                
                colors[vertexIndex * 3 + 0] = color.r;
                colors[vertexIndex * 3 + 1] = color.g;
                colors[vertexIndex * 3 + 2] = color.b;
                
                vertexIndex++;
            }
        }
        
        geometry.attributes.color.needsUpdate = true;
    }
    
    /**
     * Update 3D colors - for SDF geometry, use X-position based gradient
     * (matching the solution gradient from inlet to outlet)
     */
    updateMesh3DColors() {
        if (!this.mesh3D || !this.solutionData) return;
        
        const geometry = this.mesh3D.geometry;
        const positions = geometry.attributes.position;
        const colors = geometry.attributes.color.array;
        
        const { xMin, xMax } = this.originalBounds;
        const xRange = xMax - xMin;
        const [min, max] = this.solutionData.range;
        
        // For SDF geometry, color based on X position (approximates solution)
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const t = (x - xMin) / xRange;
            // Map X position to solution range (inlet=high, outlet=low)
            const normalized = 1 - t;
            const color = this.valueToColor(normalized);
            
            colors[i * 3 + 0] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        
        geometry.attributes.color.needsUpdate = true;
    }
    
    // =========================================================================
    // Legacy Methods
    // =========================================================================
    
    create2DMesh() {
        this.create2DGeometry();
        if (this.solutionData) this.updateMesh2DColors();
    }
    
    create3DExtrusion() {
        this.create3DGeometry();
        if (this.solutionData) this.updateMesh3DColors();
    }
    
    createAll() {
        this.create2DMesh();
        this.create3DExtrusion();
        this.geometryCreated = true;
    }
    
    // =========================================================================
    // 2D Geometry (unchanged - full mesh for reference)
    // =========================================================================
    
    createQuad8Geometry2D() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const vertices = [];
        const indices = [];
        let vertexIndex = 0;
        
        for (const elem of conn) {
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(coords.x[nodeId], coords.y[nodeId], 0);
            }
            
            indices.push(
                vertexIndex + 0, vertexIndex + 1, vertexIndex + 7,
                vertexIndex + 1, vertexIndex + 2, vertexIndex + 3,
                vertexIndex + 3, vertexIndex + 4, vertexIndex + 5,
                vertexIndex + 5, vertexIndex + 6, vertexIndex + 7,
                vertexIndex + 1, vertexIndex + 3, vertexIndex + 7,
                vertexIndex + 3, vertexIndex + 5, vertexIndex + 7
            );
            
            vertexIndex += 8;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        return geometry;
    }
    
    // =========================================================================
    // Helpers
    // =========================================================================
    
    valueToColor(t) {
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
    
    fitMeshToView(mesh) {
        const { xMin, xMax, yMin, yMax, zMin, zMax } = this.bounds;
        
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const maxXY = Math.max(xRange, yRange);
        
        const targetSize = 50;
        const scale = targetSize / maxXY;
        
        mesh.scale.set(scale, scale, scale);
        
        const centerX = (xMin + xMax) / 2;
        const centerZ = (zMin + zMax) / 2;
        
        mesh.position.set(-centerX * scale, -yMin * scale, -centerZ * scale);
    }
    
    // =========================================================================
    // Visibility Controls
    // =========================================================================
    
    set2DMeshVisible(visible) {
        this.config.show2DMesh = visible;
        if (this.mesh2D) this.mesh2D.visible = visible;
    }
    
    set3DExtrusionVisible(visible) {
        this.config.show3DExtrusion = visible;
        if (this.mesh3D) this.mesh3D.visible = visible;
    }
    
    setVisualizationMode(mode) {
        switch (mode) {
            case '2d':
                this.set2DMeshVisible(true);
                this.set3DExtrusionVisible(false);
                break;
            case '3d':
                this.set2DMeshVisible(false);
                this.set3DExtrusionVisible(true);
                break;
            case 'both':
                this.set2DMeshVisible(true);
                this.set3DExtrusionVisible(true);
                break;
        }
    }
    
    // =========================================================================
    // Particle Flow Compatibility
    // =========================================================================
    
    getYSegmentsAtXCached(x) {
        // Use the segment cache we built (same as getYSegmentsAtX)
        return this.getYSegmentsAtX(x);
    }
    
    isInsideTube(x, y, z) {
        const { xMin, xMax, yMin, yMax, zMin, zMax } = this.bounds;
        
        return (
            x >= xMin && x <= xMax &&
            y >= yMin && y <= yMax &&
            z >= zMin && z <= zMax
        );
    }
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    dispose() {
        this.scene.remove(this.group);
        
        if (this.mesh2D) {
            this.mesh2D.geometry.dispose();
            this.mesh2D.material.dispose();
        }
        
        if (this.mesh3D) {
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
    }
    
    // =========================================================================
    // Real-time Appearance Controls
    // =========================================================================
    
    setBrightness(value) {
        const v = Math.max(0, Math.min(1, value));
        if (this.mesh3D?.material) this.mesh3D.material.color.setRGB(v, v, v);
        if (this.mesh2D?.material) this.mesh2D.material.color.setRGB(v, v, v);
    }
    
    setOpacity(value) {
        if (this.mesh3D?.material) {
            this.mesh3D.material.opacity = Math.max(0, Math.min(1, value));
        }
    }
    
    setTransparent(enabled) {
        if (this.mesh3D?.material) {
            this.mesh3D.material.transparent = enabled;
        }
    }
    
    getAppearanceSettings() {
        if (!this.mesh3D?.material) return null;
        return {
            brightness: this.mesh3D.material.color.r,
            opacity: this.mesh3D.material.opacity,
            transparent: this.mesh3D.material.transparent
        };
    }
}

