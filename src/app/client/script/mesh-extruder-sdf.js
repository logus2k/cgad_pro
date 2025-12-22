import * as THREE from '../library/three.module.min.js';

/**
 * SDF-based Mesh Extruder using Marching Cubes
 * Creates 3D tube from 2D mesh with proper branch handling
 * 
 * Supports incremental color updates during solve (for 2D mesh).
 * 3D tube colors are based on X position (gradient), not solution values.
 */
export class MeshExtruderSDF {
    constructor(scene, meshData, solutionData = null, config = {}) {
        this.scene = scene;
        this.meshData = meshData;
        this.solutionData = solutionData;  // Can be null for early creation
        
        this.config = {
            show2DMesh: true,
            show3DExtrusion: false,
            resolution: [100, 60, 60],
            tubeOpacity: 0.8,
            gapThreshold: 0.15,
            ...config
        };
        
        this.mesh2D = null;
        this.mesh3D = null;
        this.group = new THREE.Group();
        
        // Track if geometry is created
        this.geometryCreated = false;
        
        // Step 1: Calculate original XY bounds (for positioning)
        this.originalBounds = this.calculateOriginalBounds();
        
        // Step 2: Calculate expanded bounds (for marching cubes)
        this.bounds = this.calculateExpandedBounds();
        
        // Step 3: Build segment cache (needs bounds)
        this.buildSegmentCache();
        
        // Step 4: Calculate Z bounds (needs segment cache)
        const zBounds = this.calculateZBounds();
        this.bounds.zMin = zBounds.zMin;
        this.bounds.zMax = zBounds.zMax;
        this.bounds.maxRadius = zBounds.maxRadius;
        
        this.scene.add(this.group);
        
        console.log('MeshExtruderSDF initialized');
    }
    
    /**
     * Calculate original XY bounds (no margin) - for positioning
     */
    calculateOriginalBounds() {
        const coords = this.meshData.coordinates;
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        for (let i = 0; i < coords.x.length; i++) {
            if (coords.x[i] < xMin) xMin = coords.x[i];
            if (coords.x[i] > xMax) xMax = coords.x[i];
            if (coords.y[i] < yMin) yMin = coords.y[i];
            if (coords.y[i] > yMax) yMax = coords.y[i];
        }

        console.log(`   Original bounds: X[${xMin.toFixed(3)}, ${xMax.toFixed(3)}], Y[${yMin.toFixed(3)}, ${yMax.toFixed(3)}]`);

        return { xMin, xMax, yMin, yMax };
    }
    
    /**
     * Calculate expanded bounds (with margin for marching cubes)
     */
    calculateExpandedBounds() {
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        
        const margin = 0.02;
        
        return {
            xMin: xMin - xRange * margin,
            xMax: xMax + xRange * margin,
            yMin: yMin - yRange * margin,
            yMax: yMax + yRange * margin
        };
    }
    
    /**
     * Calculate Z bounds based on max tube radius
     */
    calculateZBounds() {
        let maxRadius = 0;
        
        for (const entry of Object.values(this.segmentCache)) {
            if (entry && entry.segments) {
                for (const seg of entry.segments) {
                    if (seg.radius > maxRadius) {
                        maxRadius = seg.radius;
                    }
                }
            }
        }
        
        maxRadius = maxRadius || 0.5;
        const zMargin = maxRadius * 0.1;
        
        console.log(`   Max radius: ${maxRadius.toFixed(3)}, Z range: [${(-maxRadius - zMargin).toFixed(3)}, ${(maxRadius + zMargin).toFixed(3)}]`);
        
        return {
            zMin: -maxRadius - zMargin,
            zMax: maxRadius + zMargin,
            maxRadius
        };
    }
    
    /**
     * Build segment cache for fast Y-range lookup at each X
     */
    buildSegmentCache() {
        const { xMin, xMax, yMin, yMax } = this.bounds;
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const numSamples = 200;
        this.segmentCache = {};
        
        for (let i = 0; i <= numSamples; i++) {
            const x = xMin + (i / numSamples) * (xMax - xMin);
            this.segmentCache[i] = this.computeYSegmentsAtX(x, coords, conn);
        }
        
        this.segmentCacheXMin = xMin;
        this.segmentCacheXMax = xMax;
        this.segmentCacheNum = numSamples;
        
        console.log(`   Segment cache built: ${numSamples} samples`);
    }
    
    /**
     * Get Y segments at X position from cache
     */
    getYSegmentsAtXCached(x) {
        const { xMin, xMax } = this.originalBounds;
        
        if (x < xMin || x > xMax) {
            return null;
        }
        
        const t = (x - this.segmentCacheXMin) / (this.segmentCacheXMax - this.segmentCacheXMin);
        const i = Math.round(t * this.segmentCacheNum);
        const clampedI = Math.max(0, Math.min(this.segmentCacheNum, i));
        
        const entry = this.segmentCache[clampedI];
        return entry ? entry.segments : null;
    }
    
    /**
     * Compute Y segments at a given X position
     */
    computeYSegmentsAtX(x, coords, conn) {
        const yValues = [];
        const tolerance = 0.001;
        
        for (let elem of conn) {
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
        
        yValues.sort((a, b) => a.min - b.min);
        
        const merged = [];
        let current = { ...yValues[0] };
        
        for (let i = 1; i < yValues.length; i++) {
            const next = yValues[i];
            if (next.min <= current.max + this.config.gapThreshold) {
                current.max = Math.max(current.max, next.max);
            } else {
                merged.push(current);
                current = { ...next };
            }
        }
        merged.push(current);
        
        const segments = merged.map(seg => ({
            yMin: seg.min,
            yMax: seg.max,
            centerY: (seg.min + seg.max) / 2,
            radius: (seg.max - seg.min) / 2
        }));
        
        return { segments };
    }
    
    // =========================================================================
    // Geometry Creation (can be called early, before solution)
    // =========================================================================
    
    /**
     * Create geometry only (no solution colors) - for early visualization
     */
    async createGeometryOnly() {
        this.create2DGeometry();
        await this.create3DGeometry();
        this.geometryCreated = true;
        console.log('Geometry created (awaiting solution for colors)');
    }
    
    /**
     * Create 2D mesh geometry without solution colors
     */
    create2DGeometry() {
        if (this.mesh2D) {
            this.group.remove(this.mesh2D);
            this.mesh2D.geometry.dispose();
            this.mesh2D.material.dispose();
        }
        
        const geometry = this.createQuad8Geometry();
        
        // Apply neutral gray color initially
        this.applyNeutralColors(geometry);
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
        });
        
        this.mesh2D = new THREE.Mesh(geometry, material);
        this.mesh2D.castShadow = true;
        this.mesh2D.visible = this.config.show2DMesh;
        
        this.fitMeshToView(this.mesh2D);
        this.group.add(this.mesh2D);
        
        console.log('2D geometry created');
    }
    
    /**
     * Create 3D tube geometry (uses gradient colors, not solution)
     */
    async create3DGeometry() {
        if (this.mesh3D) {
            this.group.remove(this.mesh3D);
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
        
        console.log('Creating 3D tube via Marching Cubes...');
        
        const iso = await this.loadIsosurface();
        
        const { xMin, xMax, yMin, yMax, zMin, zMax } = this.bounds;
        const [resX, resY, resZ] = this.config.resolution;
        
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const zRange = zMax - zMin;
        
        const sdfFunc = (gx, gy, gz) => {
            const x = xMin + (gx / resX) * xRange;
            const y = yMin + (gy / resY) * yRange;
            const z = zMin + (gz / resZ) * zRange;
            return this.sdf(x, y, z);
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
        
        const positions = new Float32Array(result.positions.length * 3);
        for (let i = 0; i < result.positions.length; i++) {
            const [gx, gy, gz] = result.positions[i];
            positions[i * 3 + 0] = xMin + (gx / resX) * xRange;
            positions[i * 3 + 1] = yMin + (gy / resY) * yRange;
            positions[i * 3 + 2] = zMin + (gz / resZ) * zRange;
        }
        
        let indices = new Uint32Array(result.cells.length * 3);
        for (let i = 0; i < result.cells.length; i++) {
            indices[i * 3 + 0] = result.cells[i][0];
            indices[i * 3 + 1] = result.cells[i][1];
            indices[i * 3 + 2] = result.cells[i][2];
        }
        
        // =====================================================================
        // Correct MC geometry bounds BEFORE removing end caps
        // MC produces geometry slightly smaller than intended bounds
        // =====================================================================
        let geoXMin = Infinity, geoXMax = -Infinity;
        let geoYMin = Infinity, geoYMax = -Infinity;
        for (let i = 0; i < positions.length; i += 3) {
            if (positions[i] < geoXMin) geoXMin = positions[i];
            if (positions[i] > geoXMax) geoXMax = positions[i];
            if (positions[i + 1] < geoYMin) geoYMin = positions[i + 1];
            if (positions[i + 1] > geoYMax) geoYMax = positions[i + 1];
        }
        
        const origBounds = this.originalBounds;
        const geoWidth = geoXMax - geoXMin;
        const geoHeight = geoYMax - geoYMin;
        const intendedWidth = origBounds.xMax - origBounds.xMin;
        const intendedHeight = origBounds.yMax - origBounds.yMin;
        
        const scaleCorrectX = intendedWidth / geoWidth;
        const scaleCorrectY = intendedHeight / geoHeight;
        
        // Stretch vertices to match intended bounds
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] = origBounds.xMin + (positions[i] - geoXMin) * scaleCorrectX;
            positions[i + 1] = origBounds.yMin + (positions[i + 1] - geoYMin) * scaleCorrectY;
        }
        
        console.log(`   MC bounds correction: X[${geoXMin.toFixed(4)}, ${geoXMax.toFixed(4)}] -> [${origBounds.xMin}, ${origBounds.xMax}]`);
        
        // Remove end caps (inlet and outlet) - AFTER bounds correction
        indices = this.removeEndCaps(positions, indices);
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        // Apply gradient colors (based on X position)
        this.addSolutionColorsToGeometry(geometry);
        
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: this.config.tubeOpacity,
            shininess: 30
        });
        
        this.mesh3D = new THREE.Mesh(geometry, material);
        this.mesh3D.castShadow = true;
        this.mesh3D.receiveShadow = true;
        this.mesh3D.visible = this.config.show3DExtrusion;
        
        this.fitMeshToView(this.mesh3D);
        this.group.add(this.mesh3D);
        
        console.log('3D tube geometry created');
    }
    
    /**
     * Apply neutral gray color to geometry
     */
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
    // Color Updates (can be called incrementally during solve)
    // =========================================================================
    
    /**
     * Update colors based on solution data
     * @param {Object} solutionData - { values: [], range: [min, max] }
     */
    updateSolutionColors(solutionData) {
        this.solutionData = solutionData;
        
        if (this.mesh2D) {
            this.updateMesh2DColors();
        }
        
        // Note: 3D tube uses position-based gradient, not solution values
    }
    
    /**
     * Update colors for incremental solution (compatible with solution_increment event)
     * @param {Object} updateData - { solution_values: [], chunk_info: { min, max } }
     */
    updateSolutionIncremental(updateData) {
        const { solution_values, chunk_info } = updateData;
        
        this.solutionData = {
            values: solution_values,
            range: [chunk_info.min, chunk_info.max]
        };
        
        if (this.mesh2D) {
            this.updateMesh2DColors();
        }
    }
    
    /**
     * Update 2D mesh colors from current solution data
     */
    updateMesh2DColors() {
        if (!this.mesh2D || !this.solutionData) return;
        
        const geometry = this.mesh2D.geometry;
        const colors = geometry.attributes.color.array;
        
        const values = this.solutionData.values;
        const [min, max] = this.solutionData.range;
        const conn = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of conn) {
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
    
    // =========================================================================
    // SDF and Marching Cubes
    // =========================================================================
    
    sdf(x, y, z) {
        const segments = this.getYSegmentsAtXCached(x);
        
        if (!segments || segments.length === 0) {
            return 1.0;
        }
        
        let minDist = Infinity;
        
        for (const seg of segments) {
            const dy = y - seg.centerY;
            const radialDist = Math.sqrt(dy * dy + z * z);
            const dist = radialDist - seg.radius;
            
            if (dist < minDist) {
                minDist = dist;
            }
        }
        
        return minDist;
    }
    
    /**
     * Remove triangles at inlet (xMin) and outlet (xMax)
     * Detects flat end cap faces by checking if triangle is perpendicular to X axis
     */
    removeEndCaps(positions, indices) {
        const { xMin, xMax } = this.originalBounds;
        // Distance from boundary to consider as "at the end"
        const endDistance = 0.02;
        // Max X spread within a triangle to be considered a flat cap
        const flatThreshold = 0.01;
        
        const newIndices = [];
        let removedInlet = 0, removedOutlet = 0;
        
        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i + 0];
            const i1 = indices[i + 1];
            const i2 = indices[i + 2];
            
            const x0 = positions[i0 * 3 + 0];
            const x1 = positions[i1 * 3 + 0];
            const x2 = positions[i2 * 3 + 0];
            
            // Check if triangle is flat (all vertices have similar X)
            const xSpread = Math.max(x0, x1, x2) - Math.min(x0, x1, x2);
            const isFlat = xSpread < flatThreshold;
            
            // Check if at inlet or outlet
            const avgX = (x0 + x1 + x2) / 3;
            const atInlet = avgX < xMin + endDistance && isFlat;
            const atOutlet = avgX > xMax - endDistance && isFlat;
            
            if (atInlet) {
                removedInlet++;
            } else if (atOutlet) {
                removedOutlet++;
            } else {
                newIndices.push(i0, i1, i2);
            }
        }
        
        console.log(`   Removed end caps: inlet=${removedInlet}, outlet=${removedOutlet} triangles`);
        
        return new Uint32Array(newIndices);
    }
    
    async loadIsosurface() {
        if (window.isosurface) {
            return window.isosurface;
        }
        
        const module = await import('https://esm.sh/isosurface@1.0.0');
        window.isosurface = module;
        return module;
    }
    
    addSolutionColorsToGeometry(geometry) {
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        const { xMin, xMax } = this.originalBounds;
        
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            
            const normalized = 1 - (x - xMin) / (xMax - xMin);
            const color = this.valueToColor(normalized);
            
            colors[i * 3 + 0] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    // =========================================================================
    // Legacy Methods (for backward compatibility)
    // =========================================================================
    
    /**
     * Create 2D mesh - legacy method
     */
    create2DMesh() {
        this.create2DGeometry();
        if (this.solutionData) {
            this.updateMesh2DColors();
        }
    }
    
    /**
     * Create 3D extrusion - legacy method
     */
    async create3DExtrusion() {
        await this.create3DGeometry();
    }
    
    /**
     * Create all - legacy method
     */
    async createAll() {
        this.create2DMesh();
        await this.create3DExtrusion();
        this.geometryCreated = true;
    }
    
    // =========================================================================
    // Helper Methods
    // =========================================================================
    
    createQuad8Geometry() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const vertices = [];
        const indices = [];
        let vertexIndex = 0;
        
        for (let elem of conn) {
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
    
    addSolutionColors(geometry) {
        if (!this.solutionData) {
            this.applyNeutralColors(geometry);
            return;
        }
        
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        const values = this.solutionData.values;
        const [min, max] = this.solutionData.range;
        const conn = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of conn) {
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                const value = values[nodeId];
                const normalized = (value - min) / (max - min);
                const color = this.valueToColor(normalized);
                
                colors[vertexIndex * 3 + 0] = color.r;
                colors[vertexIndex * 3 + 1] = color.g;
                colors[vertexIndex * 3 + 2] = color.b;
                
                vertexIndex++;
            }
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
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
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        const { zMin, zMax } = this.bounds;
        
        const sizeX = xMax - xMin;
        const sizeY = yMax - yMin;
        const maxDim = Math.max(sizeX, sizeY);
        const targetSize = 50;
        const scale = targetSize / maxDim;
        
        // Store scale for camera controller
        this.scale = scale;
        
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
            default:
                console.warn(`Unknown visualization mode: ${mode}`);
        }
        console.log(`Visualization mode: ${mode}`);
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
    // Real-time Appearance Controls (no regeneration needed)
    // =========================================================================
    
    /**
     * Set brightness of the 3D mesh (multiplies vertex colors)
     * @param {number} value - 0.0 (black) to 1.0 (full bright), default 1.0
     */
    setBrightness(value) {
        if (this.mesh3D && this.mesh3D.material) {
            const v = Math.max(0, Math.min(1, value));
            this.mesh3D.material.color.setRGB(v, v, v);
        }
        if (this.mesh2D && this.mesh2D.material) {
            const v = Math.max(0, Math.min(1, value));
            this.mesh2D.material.color.setRGB(v, v, v);
        }
    }
    
    /**
     * Set opacity of the 3D mesh
     * @param {number} value - 0.0 (invisible) to 1.0 (fully opaque), default 0.8
     */
    setOpacity(value) {
        if (this.mesh3D && this.mesh3D.material) {
            this.mesh3D.material.opacity = Math.max(0, Math.min(1, value));
        }
    }
    
    /**
     * Enable or disable transparency
     * @param {boolean} enabled - true to enable transparency, false to disable
     */
    setTransparent(enabled) {
        if (this.mesh3D && this.mesh3D.material) {
            this.mesh3D.material.transparent = enabled;
        }
    }
    
    /**
     * Get current appearance settings
     * @returns {Object} Current brightness, opacity, and transparency values
     */
    getAppearanceSettings() {
        if (!this.mesh3D || !this.mesh3D.material) {
            return null;
        }
        return {
            brightness: this.mesh3D.material.color.r,
            opacity: this.mesh3D.material.opacity,
            transparent: this.mesh3D.material.transparent
        };
    }
}
