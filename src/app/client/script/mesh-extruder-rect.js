import * as THREE from '../library/three.module.min.js';

/**
 * MeshExtruderRect - Rectangular extrusion for 2D FEM meshes
 * 
 * Creates a 3D slab by extruding the 2D mesh along the Z axis.
 * Standard FEM visualization where Z represents "infinite" extent.
 * 
 * Z >> max(X, Y) to represent plane strain/plane flow assumption.
 * 
 * Supports incremental color updates during solve.
 */
export class MeshExtruderRect {
    constructor(scene, meshData, solutionData = null, config = {}) {
        this.scene = scene;
        this.meshData = meshData;
        this.solutionData = solutionData;  // Can be null for early creation
        
        this.config = {
            show2DMesh: true,
            show3DExtrusion: false,
            zFactor: 5.0,           // Z = zFactor * max(X, Y)
            extrusionOpacity: 0.8,
            ...config
        };
        
        this.mesh2D = null;
        this.mesh3D = null;
        this.group = new THREE.Group();
        
        // Track if geometry is created (for incremental updates)
        this.geometryCreated = false;
        
        // Calculate bounds
        this.bounds = this.calculateBounds();
        
        // Store original bounds for particle flow compatibility
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
        
        // Z >> max(X, Y) for "infinite" extent representation
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
    
    // =========================================================================
    // Geometry Creation (can be called early, before solution)
    // =========================================================================
    
    /**
     * Create geometry only (no colors) - for early visualization
     */
    createGeometryOnly() {
        this.create2DGeometry();
        this.create3DGeometry();
        this.geometryCreated = true;
        console.log('Geometry created (awaiting solution for colors)');
    }
    
    /**
     * Create 2D mesh geometry without colors
     */
    create2DGeometry() {
        if (this.mesh2D) {
            this.group.remove(this.mesh2D);
            this.mesh2D.geometry.dispose();
            this.mesh2D.material.dispose();
        }
        
        const geometry = this.createQuad8Geometry2D();
        
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
     * Create 3D extruded geometry without colors
     */
    create3DGeometry() {
        if (this.mesh3D) {
            this.group.remove(this.mesh3D);
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
        
        console.log('Creating 3D rectangular geometry...');
        
        const geometry = this.createExtrudedGeometry();
        
        // Apply neutral gray color initially
        this.applyNeutralColors(geometry);
        
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: this.config.extrusionOpacity,
            shininess: 30
        });
        
        this.mesh3D = new THREE.Mesh(geometry, material);
        this.mesh3D.castShadow = true;
        this.mesh3D.receiveShadow = true;
        this.mesh3D.visible = this.config.show3DExtrusion;
        
        this.fitMeshToView(this.mesh3D);
        this.group.add(this.mesh3D);
        
        console.log('3D geometry created');
    }
    
    /**
     * Apply neutral gray color to geometry (before solution is available)
     */
    applyNeutralColors(geometry) {
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        // Light gray color
        for (let i = 0; i < positions.count; i++) {
            colors[i * 3 + 0] = 0.75;  // R
            colors[i * 3 + 1] = 0.75;  // G
            colors[i * 3 + 2] = 0.75;  // B
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    // =========================================================================
    // Color Updates (can be called incrementally during solve)
    // =========================================================================
    
    /**
     * Update colors based on solution data (incremental update)
     * @param {Object} solutionData - { values: [], range: [min, max] }
     */
    updateSolutionColors(solutionData) {
        this.solutionData = solutionData;
        
        if (this.mesh2D) {
            this.updateMesh2DColors();
        }
        
        if (this.mesh3D) {
            this.updateMesh3DColors();
        }
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
        
        if (this.mesh3D) {
            this.updateMesh3DColors();
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
    
    /**
     * Update 3D mesh colors from current solution data
     */
    updateMesh3DColors() {
        if (!this.mesh3D || !this.solutionData) return;
        
        const geometry = this.mesh3D.geometry;
        const colors = geometry.attributes.color.array;
        
        const values = this.solutionData.values;
        const [min, max] = this.solutionData.range;
        const conn = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of conn) {
            // Front face (8 vertices) + Back face (8 vertices)
            for (let face = 0; face < 2; face++) {
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
        }
        
        geometry.attributes.color.needsUpdate = true;
    }
    
    // =========================================================================
    // Legacy Methods (for backward compatibility)
    // =========================================================================
    
    /**
     * Create 2D mesh visualization (on XY plane) - legacy method
     */
    create2DMesh() {
        this.create2DGeometry();
        if (this.solutionData) {
            this.updateMesh2DColors();
        }
    }
    
    /**
     * Create 3D extruded mesh (rectangular slab) - legacy method
     */
    create3DExtrusion() {
        this.create3DGeometry();
        if (this.solutionData) {
            this.updateMesh3DColors();
        }
    }
    
    /**
     * Create all visualizations - legacy method
     */
    createAll() {
        this.create2DMesh();
        this.create3DExtrusion();
        this.geometryCreated = true;
    }
    
    // =========================================================================
    // Geometry Creation Helpers
    // =========================================================================
    
    /**
     * Create Quad-8 geometry for 2D visualization
     */
    createQuad8Geometry2D() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const vertices = [];
        const indices = [];
        let vertexIndex = 0;
        
        for (let elem of conn) {
            // Add all 8 vertices for this element (Z = 0)
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(coords.x[nodeId], coords.y[nodeId], 0);
            }
            
            // 6-triangle subdivision for Quad-8
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
    
    /**
     * Create extruded geometry (front face, back face, connected as solid)
     */
    createExtrudedGeometry() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        const { zMin, zMax } = this.bounds;
        
        const vertices = [];
        const indices = [];
        let vertexIndex = 0;
        
        for (let elem of conn) {
            // Front face (Z = zMax) - 8 vertices
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(coords.x[nodeId], coords.y[nodeId], zMax);
            }
            
            // Back face (Z = zMin) - 8 vertices
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(coords.x[nodeId], coords.y[nodeId], zMin);
            }
            
            const front = vertexIndex;      // Vertices 0-7: front face
            const back = vertexIndex + 8;   // Vertices 8-15: back face
            
            // Front face triangles
            indices.push(
                front + 0, front + 1, front + 7,
                front + 1, front + 2, front + 3,
                front + 3, front + 4, front + 5,
                front + 5, front + 6, front + 7,
                front + 1, front + 3, front + 7,
                front + 3, front + 5, front + 7
            );
            
            // Back face triangles (reversed winding)
            indices.push(
                back + 0, back + 7, back + 1,
                back + 1, back + 3, back + 2,
                back + 3, back + 5, back + 4,
                back + 5, back + 7, back + 6,
                back + 1, back + 7, back + 3,
                back + 3, back + 7, back + 5
            );
            
            // Side faces connecting front and back
            indices.push(front + 0, back + 0, front + 1);
            indices.push(front + 1, back + 0, back + 1);
            indices.push(front + 1, back + 1, front + 2);
            indices.push(front + 2, back + 1, back + 2);
            indices.push(front + 2, back + 2, front + 3);
            indices.push(front + 3, back + 2, back + 3);
            indices.push(front + 3, back + 3, front + 4);
            indices.push(front + 4, back + 3, back + 4);
            indices.push(front + 4, back + 4, front + 5);
            indices.push(front + 5, back + 4, back + 5);
            indices.push(front + 5, back + 5, front + 6);
            indices.push(front + 6, back + 5, back + 6);
            indices.push(front + 6, back + 6, front + 7);
            indices.push(front + 7, back + 6, back + 7);
            indices.push(front + 7, back + 7, front + 0);
            indices.push(front + 0, back + 7, back + 0);
            
            vertexIndex += 16;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        return geometry;
    }
    
    /**
     * Color map: blue (low) -> cyan -> green -> yellow -> red (high)
     */
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
    
    /**
     * Fit mesh to view - scale based on X/Y only
     */
    fitMeshToView(mesh) {
        const { xMin, xMax, yMin, yMax, zMin, zMax } = this.bounds;
        
        // Scale based on X/Y dimensions only (not Z)
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const maxXY = Math.max(xRange, yRange);
        
        const targetSize = 50;
        const scale = targetSize / maxXY;
        
        mesh.scale.set(scale, scale, scale);
        
        const centerX = (xMin + xMax) / 2;
        const centerZ = (zMin + zMax) / 2;
        
        // Center horizontally, rest on floor (Y), center in Z
        mesh.position.set(-centerX * scale, -yMin * scale, -centerZ * scale);
        
        console.log(`   Fitted to view: scale=${scale.toFixed(4)} (based on X/Y only)`);
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
    
    /**
     * Set visualization mode
     * @param {string} mode - '2d', '3d', or 'both'
     */
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
    // Particle Flow Compatibility
    // =========================================================================
    
    /**
     * Get Y segments at X position (for particle flow compatibility)
     */
    getYSegmentsAtXCached(x) {
        const { xMin, xMax, yMin, yMax } = this.bounds;
        
        if (x < xMin || x > xMax) {
            return null;
        }
        
        return [{
            yMin: yMin,
            yMax: yMax,
            centerY: (yMin + yMax) / 2,
            radius: (yMax - yMin) / 2
        }];
    }
    
    /**
     * Check if point is inside the extruded volume
     */
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
}
