import * as THREE from '../library/three.module.min.js';

/**
 * MeshExtruderRect - Rectangular extrusion for 2D FEM meshes
 * 
 * Creates a hollow 3D tube by extruding only the BOUNDARY edges of the 2D mesh.
 * Inlet (xMin) and outlets (xMax) are left open for particle flow.
 * 
 * Supports incremental color updates during solve with proper solution interpolation.
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
        this.wallNodeMap = null;
        
        // Solution interpolation mapping (MC vertex -> FEM element + local coords)
        this.vertexElementMap = null;
        this.elementGrid = null;
        
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
    
    buildYRangeCache() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        const { xMin, xMax } = this.originalBounds;
        
        this.cacheResolution = 500;
        this.ySegmentCache = new Array(this.cacheResolution);
        
        const xRange = xMax - xMin;
        const gapThreshold = 0.05;
        
        console.log('   Building Y-segment cache for rectangular SDF...');
        
        for (let i = 0; i < this.cacheResolution; i++) {
            const x = xMin + (i / (this.cacheResolution - 1)) * xRange;
            this.ySegmentCache[i] = this.computeYSegmentsAtX(x, coords, conn, gapThreshold);
        }
        
        const midIdx = Math.floor(this.cacheResolution / 2);
        const endIdx = this.cacheResolution - 10;
        console.log(`   At inlet: ${this.ySegmentCache[10]?.length || 0} segments`);
        console.log(`   At middle: ${this.ySegmentCache[midIdx]?.length || 0} segments`);
        console.log(`   At outlet: ${this.ySegmentCache[endIdx]?.length || 0} segments`);
    }
    
    computeYSegmentsAtX(x, coords, conn, gapThreshold) {
        const yValues = [];
        const tolerance = 0.001;
        
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
        
        yValues.sort((a, b) => a.min - b.min);
        
        const segments = [];
        let current = { ...yValues[0] };
        
        for (let i = 1; i < yValues.length; i++) {
            const next = yValues[i];
            if (next.min <= current.max + gapThreshold) {
                current.max = Math.max(current.max, next.max);
            } else {
                // Add segment with yMin, yMax, centerY, and radius for cylindrical mode
                segments.push({
                    yMin: current.min,
                    yMax: current.max,
                    centerY: (current.min + current.max) / 2,
                    radius: (current.max - current.min) / 2
                });
                current = { ...next };
            }
        }
        // Add final segment with all properties
        segments.push({
            yMin: current.min,
            yMax: current.max,
            centerY: (current.min + current.max) / 2,
            radius: (current.max - current.min) / 2
        });
        
        return segments;
    }
    
    getYSegmentsAtX(x) {
        const { xMin, xMax } = this.originalBounds;
        
        if (x < xMin || x > xMax) return null;
        
        const t = (x - xMin) / (xMax - xMin);
        const idx = Math.floor(t * (this.cacheResolution - 1));
        
        if (idx < 0 || idx >= this.cacheResolution) return null;
        
        return this.ySegmentCache[idx];
    }
    
    rectangularSDF(x, y, z) {
        const { xMin, xMax } = this.originalBounds;
        const { zMin, zMax } = this.bounds;
        
        if (x < xMin || x > xMax) {
            return 1.0;
        }
        
        const segments = this.getYSegmentsAtX(x);
        if (!segments || segments.length === 0) {
            return 1.0;
        }
        
        let minDist = Infinity;
        
        for (const seg of segments) {
            const { yMin, yMax } = seg;
            
            const dTop = y - yMax;
            const dBottom = yMin - y;
            const dY = Math.max(dTop, dBottom);
            
            const dFront = z - zMax;
            const dBack = zMin - z;
            const dZ = Math.max(dFront, dBack);
            
            const dist = Math.max(dY, dZ);
            
            if (dist < minDist) {
                minDist = dist;
            }
        }
        
        return minDist;
    }
    
    // =========================================================================
    // Spatial Index for FEM Elements (for solution interpolation)
    // =========================================================================
    
    buildElementGrid() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        
        const gridResX = 50;
        const gridResY = 50;
        
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const cellWidth = xRange / gridResX;
        const cellHeight = yRange / gridResY;
        
        this.elementGrid = {
            resX: gridResX,
            resY: gridResY,
            xMin, xMax, yMin, yMax,
            cellWidth, cellHeight,
            cells: new Array(gridResX * gridResY).fill(null).map(() => [])
        };
        
        console.log(`   Building element grid (${gridResX}x${gridResY})...`);
        
        for (let elemIdx = 0; elemIdx < conn.length; elemIdx++) {
            const elem = conn[elemIdx];
            
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
            
            const iMin = Math.max(0, Math.floor((elemXMin - xMin) / cellWidth));
            const iMax = Math.min(gridResX - 1, Math.floor((elemXMax - xMin) / cellWidth));
            const jMin = Math.max(0, Math.floor((elemYMin - yMin) / cellHeight));
            const jMax = Math.min(gridResY - 1, Math.floor((elemYMax - yMin) / cellHeight));
            
            for (let i = iMin; i <= iMax; i++) {
                for (let j = jMin; j <= jMax; j++) {
                    this.elementGrid.cells[j * gridResX + i].push(elemIdx);
                }
            }
        }
        
        let maxPerCell = 0, totalEntries = 0;
        for (const cell of this.elementGrid.cells) {
            if (cell.length > maxPerCell) maxPerCell = cell.length;
            totalEntries += cell.length;
        }
        console.log(`   Element grid built: max ${maxPerCell} elements/cell, avg ${(totalEntries / this.elementGrid.cells.length).toFixed(1)}`);
    }
    
    findContainingElement(x, y) {
        if (!this.elementGrid) return null;
        
        const { resX, resY, xMin, yMin, cellWidth, cellHeight, cells } = this.elementGrid;
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const i = Math.floor((x - xMin) / cellWidth);
        const j = Math.floor((y - yMin) / cellHeight);
        
        if (i < 0 || i >= resX || j < 0 || j >= resY) return null;
        
        const cellElements = cells[j * resX + i];
        
        for (const elemIdx of cellElements) {
            const elem = conn[elemIdx];
            
            const x0 = coords.x[elem[0]], y0 = coords.y[elem[0]];
            const x2 = coords.x[elem[2]], y2 = coords.y[elem[2]];
            const x4 = coords.x[elem[4]], y4 = coords.y[elem[4]];
            const x6 = coords.x[elem[6]], y6 = coords.y[elem[6]];
            
            const bxMin = Math.min(x0, x2, x4, x6);
            const bxMax = Math.max(x0, x2, x4, x6);
            const byMin = Math.min(y0, y2, y4, y6);
            const byMax = Math.max(y0, y2, y4, y6);
            
            if (x < bxMin || x > bxMax || y < byMin || y > byMax) continue;
            
            const xi = 2 * (x - (bxMin + bxMax) / 2) / (bxMax - bxMin);
            const eta = 2 * (y - (byMin + byMax) / 2) / (byMax - byMin);
            
            if (xi >= -1.1 && xi <= 1.1 && eta >= -1.1 && eta <= 1.1) {
                return { 
                    elemIdx, 
                    xi: Math.max(-1, Math.min(1, xi)), 
                    eta: Math.max(-1, Math.min(1, eta)) 
                };
            }
        }
        
        return null;
    }
    
    quad8ShapeFunctions(xi, eta) {
        const xi2 = xi * xi;
        const eta2 = eta * eta;
        
        const N0 = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1);
        const N2 = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1);
        const N4 = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1);
        const N6 = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1);
        
        const N1 = 0.5 * (1 - xi2) * (1 - eta);
        const N3 = 0.5 * (1 + xi) * (1 - eta2);
        const N5 = 0.5 * (1 - xi2) * (1 + eta);
        const N7 = 0.5 * (1 - xi) * (1 - eta2);
        
        return [N0, N1, N2, N3, N4, N5, N6, N7];
    }
    
    interpolateSolution(elemIdx, xi, eta, solutionValues) {
        const elem = this.meshData.connectivity[elemIdx];
        const N = this.quad8ShapeFunctions(xi, eta);
        
        let value = 0;
        for (let i = 0; i < 8; i++) {
            const nodeId = elem[i];
            value += N[i] * (solutionValues[nodeId] || 0);
        }
        
        return value;
    }
    
    buildVertexElementMapping(positions) {
        if (!this.elementGrid) {
            this.buildElementGrid();
        }
        
        const vertexCount = positions.count;
        this.vertexElementMap = new Array(vertexCount);
        
        let mapped = 0, unmapped = 0;
        
        console.log(`   Building vertex-element mapping for ${vertexCount} vertices...`);
        
        for (let i = 0; i < vertexCount; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            
            const result = this.findContainingElement(x, y);
            
            if (result) {
                this.vertexElementMap[i] = result;
                mapped++;
            } else {
                this.vertexElementMap[i] = null;
                unmapped++;
            }
        }
        
        console.log(`   Vertex mapping: ${mapped} mapped, ${unmapped} unmapped (${(100 * mapped / vertexCount).toFixed(1)}%)`);
    }
    
    // =========================================================================
    // Geometry Creation
    // =========================================================================
    
    createGeometryOnly() {
        this.buildYRangeCache();
        this.buildElementGrid();
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
        
        const positions = new Float32Array(result.positions.length * 3);
        for (let i = 0; i < result.positions.length; i++) {
            const [gx, gy, gz] = result.positions[i];
            positions[i * 3 + 0] = mcBounds.xMin + (gx / resX) * mcXRange;
            positions[i * 3 + 1] = mcBounds.yMin + (gy / resY) * mcYRange;
            positions[i * 3 + 2] = mcBounds.zMin + (gz / resZ) * mcZRange;
        }
        
        let indices = new Uint32Array(result.cells.length * 3);
        for (let i = 0; i < result.cells.length; i++) {
            indices[i * 3 + 0] = result.cells[i][0];
            indices[i * 3 + 1] = result.cells[i][1];
            indices[i * 3 + 2] = result.cells[i][2];
        }
        
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
        
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] = origBounds.xMin + (positions[i] - geoXMin) * scaleCorrectX;
            positions[i + 1] = origBounds.yMin + (positions[i + 1] - geoYMin) * scaleCorrectY;
        }
        
        console.log(`   MC bounds correction: X[${geoXMin.toFixed(4)}, ${geoXMax.toFixed(4)}] -> [${origBounds.xMin}, ${origBounds.xMax}]`);
        
        indices = this.removeEndCaps(positions, indices);
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        this.applyNeutralColors(geometry);
        
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
        
        this.buildVertexElementMapping(geometry.attributes.position);
    }
    
    removeEndCaps(positions, indices) {
        const { xMin, xMax } = this.originalBounds;
        const endDistance = 0.02;
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
            
            const xSpread = Math.max(x0, x1, x2) - Math.min(x0, x1, x2);
            const isFlat = xSpread < flatThreshold;
            
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
    
    addGradientColors(geometry) {
        const positions = geometry.attributes.position;
        const colors = new Float32Array(positions.count * 3);
        
        const { xMin, xMax } = this.originalBounds;
        const xRange = xMax - xMin;
        
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const t = (x - xMin) / xRange;
            const color = this.valueToColor(1 - t);
            
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
            colors[i * 3 + 0] = 0.6;
            colors[i * 3 + 1] = 0.6;
            colors[i * 3 + 2] = 0.7;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    // =========================================================================
    // Color Updates with Solution Interpolation
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
    
    updateMesh3DColors() {
        if (!this.mesh3D || !this.solutionData) return;
        
        const geometry = this.mesh3D.geometry;
        const positions = geometry.attributes.position;
        const colors = geometry.attributes.color.array;
        
        const { xMin, xMax } = this.originalBounds;
        const xRange = xMax - xMin;
        const values = this.solutionData.values;
        const [min, max] = this.solutionData.range;
        
        const hasMapping = this.vertexElementMap && this.vertexElementMap.length === positions.count;
        
        for (let i = 0; i < positions.count; i++) {
            let value;
            
            if (hasMapping && this.vertexElementMap[i]) {
                const { elemIdx, xi, eta } = this.vertexElementMap[i];
                value = this.interpolateSolution(elemIdx, xi, eta, values);
            } else {
                const x = positions.getX(i);
                const t = (x - xMin) / xRange;
                value = max * (1 - t) + min * t;
            }
            
            const normalized = (max > min) ? (value - min) / (max - min) : 0.5;
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
        
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const maxXY = Math.max(xRange, yRange);
        
        const targetSize = 50;
        const scale = targetSize / maxXY;
        
        this.scale = scale;
        
        mesh.scale.set(scale, scale, scale);
        
        const centerX = (xMin + xMax) / 2;
        const centerZ = (zMin + zMax) / 2;
        
        mesh.position.set(-centerX * scale, -yMin * scale, -centerZ * scale);
    }
    
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
    
    getYSegmentsAtXCached(x) {
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
        
        this.vertexElementMap = null;
        this.elementGrid = null;
    }
    
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
