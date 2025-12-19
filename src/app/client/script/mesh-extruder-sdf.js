import * as THREE from '../library/three.module.min.js';

/**
 * SDF-based Mesh Extruder using Marching Cubes
 * Creates 3D tube from 2D mesh with proper branch handling
 */
export class MeshExtruderSDF {
    constructor(scene, meshData, solutionData, config = {}) {
        this.scene = scene;
        this.meshData = meshData;
        this.solutionData = solutionData;
        
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
     * Calculate expanded bounds (with margin) - for marching cubes
     */
    calculateExpandedBounds() {
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        
        const margin = Math.max(xMax - xMin, yMax - yMin) * 0.05;
        
        return { 
            xMin: xMin - margin, 
            xMax: xMax + margin, 
            yMin: yMin - margin, 
            yMax: yMax + margin 
        };
    }
    
    /**
     * Calculate Z bounds based on maximum radius in segment cache
     */
    calculateZBounds() {
        let maxRadius = 0;
        
        for (const segments of this.segmentCache) {
            if (!segments) continue;
            for (const seg of segments) {
                if (seg.radius > maxRadius) {
                    maxRadius = seg.radius;
                }
            }
        }
        
        const zExtent = maxRadius * 1.1;
        
        console.log(`   Max radius found: ${maxRadius.toFixed(3)}, Z extent: +/-${zExtent.toFixed(3)}`);
        
        return { zMin: -zExtent, zMax: zExtent, maxRadius };
    }
    
    /**
     * Pre-compute Y segments at each X position
     */
    buildSegmentCache() {
        const coords = this.meshData.coordinates;
        const { xMin, xMax, yMin, yMax } = this.originalBounds;
        const yRange = yMax - yMin;
        
        const cacheResolution = this.config.resolution[0] * 2;
        this.segmentCache = [];
        this.segmentCacheResolution = cacheResolution;
        
        const dx = (xMax - xMin) / cacheResolution;
        const tolerance = dx * 1.5;
        const gapThreshold = yRange * this.config.gapThreshold;
        
        console.log(`   Building segment cache: ${cacheResolution} X positions...`);
        
        for (let i = 0; i <= cacheResolution; i++) {
            const x = xMin + i * dx;
            
            const yValues = [];
            for (let j = 0; j < coords.x.length; j += 3) {
                if (Math.abs(coords.x[j] - x) < tolerance) {
                    yValues.push(coords.y[j]);
                }
            }
            
            if (yValues.length === 0) {
                this.segmentCache[i] = null;
                continue;
            }
            
            yValues.sort((a, b) => a - b);
            
            const segments = [];
            let segStart = yValues[0];
            let segEnd = yValues[0];
            
            for (let j = 1; j < yValues.length; j++) {
                const gap = yValues[j] - yValues[j - 1];
                
                if (gap > gapThreshold) {
                    segments.push({ 
                        yMin: segStart, 
                        yMax: segEnd,
                        centerY: (segStart + segEnd) / 2,
                        radius: (segEnd - segStart) / 2
                    });
                    segStart = yValues[j];
                }
                segEnd = yValues[j];
            }
            
            segments.push({ 
                yMin: segStart, 
                yMax: segEnd,
                centerY: (segStart + segEnd) / 2,
                radius: (segEnd - segStart) / 2
            });
            
            this.segmentCache[i] = segments;
        }
        
        console.log(`   Segment cache built: ${cacheResolution + 1} entries`);
    }
    
    /**
     * Get cached Y segments at X (O(1) lookup)
     */
    getYSegmentsAtXCached(x) {
        const { xMin, xMax } = this.originalBounds;
        
        const t = (x - xMin) / (xMax - xMin);
        const idx = Math.round(t * this.segmentCacheResolution);
        
        if (idx < 0 || idx > this.segmentCacheResolution) {
            return null;
        }
        
        return this.segmentCache[idx];
    }
    
    /**
     * Signed Distance Function (FAST cached version)
     */
    sdf(x, y, z) {
        const { xMin, xMax, yMin, yMax } = this.bounds;
        
        if (x < xMin || x > xMax || y < yMin || y > yMax) {
            return 1.0;
        }
        
        const segments = this.getYSegmentsAtXCached(x);
        
        if (!segments || segments.length === 0) {
            return 1.0;
        }
        
        let bestSDF = Infinity;
        
        for (const seg of segments) {
            const distFromCenter = Math.sqrt(
                (y - seg.centerY) * (y - seg.centerY) + z * z
            );
            
            const segSDF = distFromCenter - seg.radius;
            
            if (segSDF < bestSDF) {
                bestSDF = segSDF;
            }
        }
        
        return bestSDF;
    }

    /**
     * Check if point is inside tube
     */
    isInsideTube(x, y, z) {
        return this.sdf(x, y, z) < 0;
    }

    /**
     * Remove end cap faces to open the tube ends
     */
    removeEndCaps(positions, indices) {
        let meshXMin = Infinity, meshXMax = -Infinity;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            if (x < meshXMin) meshXMin = x;
            if (x > meshXMax) meshXMax = x;
        }
        
        console.log(`   Mesh X range: [${meshXMin.toFixed(3)}, ${meshXMax.toFixed(3)}]`);
        
        const xRange = meshXMax - meshXMin;
        const tolerance = xRange * 0.015;
        
        const newIndices = [];
        let removedInlet = 0;
        let removedOutlet = 0;
        
        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i];
            const i1 = indices[i + 1];
            const i2 = indices[i + 2];
            
            const x0 = positions[i0 * 3];
            const x1 = positions[i1 * 3];
            const x2 = positions[i2 * 3];
            
            const atInlet = (
                x0 < meshXMin + tolerance &&
                x1 < meshXMin + tolerance &&
                x2 < meshXMin + tolerance
            );
            
            const atOutlet = (
                x0 > meshXMax - tolerance &&
                x1 > meshXMax - tolerance &&
                x2 > meshXMax - tolerance
            );
            
            if (atInlet) {
                removedInlet++;
            } else if (atOutlet) {
                removedOutlet++;
            } else {
                newIndices.push(i0, i1, i2);
            }
        }
        
        console.log(`   Removed ${removedInlet} inlet faces, ${removedOutlet} outlet faces`);
        
        return new Uint32Array(newIndices);
    }
    
    async create3DExtrusion() {
        if (this.mesh3D) {
            this.group.remove(this.mesh3D);
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
        
        console.log('Creating 3D tube using Marching Cubes...');
        
        const isosurface = await this.loadIsosurface();
        
        const { xMin, xMax, yMin, yMax, zMin, zMax, maxRadius } = this.bounds;
        const yRange = yMax - yMin;
        const xRange = xMax - xMin;
        const zRange = zMax - zMin;
        
        const [resX, resY, resZ] = this.config.resolution;
        
        console.log(`   Resolution: ${resX}x${resY}x${resZ}`);
        console.log(`   Bounds: X[${xMin.toFixed(2)}, ${xMax.toFixed(2)}], Y[${yMin.toFixed(2)}, ${yMax.toFixed(2)}], Z[${zMin.toFixed(2)}, ${zMax.toFixed(2)}]`);
        console.log(`   Max tube radius: ${maxRadius.toFixed(3)}`);
        
        const result = isosurface.marchingCubes(
            [resX, resY, resZ],
            (gx, gy, gz) => {
                const wx = xMin + (gx / resX) * xRange;
                const wy = yMin + (gy / resY) * yRange;
                const wz = zMin + (gz / resZ) * zRange;
                
                return this.sdf(wx, wy, wz);
            },
            [[0, 0, 0], [resX, resY, resZ]]
        );
        
        console.log(`   Marching cubes: ${result.positions.length} vertices, ${result.cells.length} faces`);
        
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
        
        indices = this.removeEndCaps(positions, indices);
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        this.addSolutionColorsToGeometry(geometry);
        
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: this.config.tubeOpacity,
            shininess: 30
        });
        
        this.mesh3D = new THREE.Mesh(geometry, material);
        this.mesh3D.visible = this.config.show3DExtrusion;
        
        this.fitMeshToView(this.mesh3D);
        this.group.add(this.mesh3D);
        
        console.log('3D tube created via Marching Cubes');
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
        
        return new THREE.Color(r, g, b);
    }
    
    create2DMesh() {
        if (this.mesh2D) {
            this.group.remove(this.mesh2D);
            this.mesh2D.geometry.dispose();
            this.mesh2D.material.dispose();
        }
        
        const geometry = this.createQuad8Geometry();
        this.addSolutionColors(geometry);
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
        });
        
        this.mesh2D = new THREE.Mesh(geometry, material);
        this.mesh2D.visible = this.config.show2DMesh;
        
        this.fitMeshToView(this.mesh2D);
        this.group.add(this.mesh2D);
        
        console.log('2D mesh created');
    }
    
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
    
    fitMeshToView(mesh) {
        const { xMin, xMax, yMin, yMax } = this.originalBounds;

        const sizeX = xMax - xMin;
        const sizeY = yMax - yMin;
        const maxDim = Math.max(sizeX, sizeY);
        const targetSize = 50;
        const scale = targetSize / maxDim;

        mesh.scale.set(scale, scale, scale);

        const centerX = (xMin + xMax) / 2;
        mesh.position.set(-centerX * scale, -yMin * scale, 0);

        console.log(`   Fitted to view: scale=${scale.toFixed(4)}`);
    }
    
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
    
    async createAll() {
        this.create2DMesh();
        await this.create3DExtrusion();
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
    }
}
