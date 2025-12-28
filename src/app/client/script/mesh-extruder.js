import * as THREE from '../library/three.module.min.js';

/**
 * Mesh Extruder - Reconstructs 3D tube from 2D longitudinal cut
 * 
 * Handles:
 * - Variable radius along tube length
 * - Y-bifurcations (single inlet → multiple outlets)
 * - Circular cross-sections
 */
export class MeshExtruder {
    constructor(scene, meshData, solutionData, config = {}) {
        this.scene = scene;
        this.meshData = meshData;
        this.solutionData = solutionData;
        
        // Configuration with defaults
        this.config = {
            show2DMesh: true,           // Show original 2D mesh
            show3DExtrusion: false,     // Show 3D tube
            radialSegments: 32,         // Smoothness around circumference
            extrusionSamples: 200,      // Number of cross-sections along length
            tubeOpacity: 0.7,           // Transparency
            branchGapThreshold: 0.3,    // Y-gap to detect separate branches (relative)
            ...config
        };
        
        this.mesh2D = null;
        this.mesh3D = null;
        this.group = new THREE.Group();
        
        this.bounds = this.calculateBounds();
        this.scene.add(this.group);
        
        console.log('OK: MeshExtruder initialized');
    }
    
    /**
     * Calculate mesh bounds
     */
    calculateBounds() {
        const coords = this.meshData.coordinates;
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        for (let i = 0; i < coords.x.length; i++) {
            const x = coords.x[i];
            const y = coords.y[i];
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }

        return { xMin, xMax, yMin, yMax };
    }
    
    /**
     * Create the 2D mesh (original visualization)
     */
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
            side: THREE.DoubleSide,
            wireframe: false
        });
        
        this.mesh2D = new THREE.Mesh(geometry, material);
        this.mesh2D.visible = this.config.show2DMesh;
        
        this.fitMeshToView(this.mesh2D);
        
        this.group.add(this.mesh2D);
        
        console.log('OK: 2D mesh created');
    }
    
    /**
     * Create 3D tube with variable radius and bifurcation support
     */
    create3DExtrusion() {
        if (this.mesh3D) {
            this.group.remove(this.mesh3D);
            this.mesh3D.geometry.dispose();
            this.mesh3D.material.dispose();
        }
        
        console.log('🌊 Creating Y-shaped tube with variable radius...');
        
        const geometry = this.createBranchedTubeGeometry();
        
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
        
        console.log('OK: 3D Y-shaped tube created');
    }
    
    /**
     * Create tube geometry with branch detection
     */
    createBranchedTubeGeometry() {
        const coords = this.meshData.coordinates;
        const { xMin, xMax, yMin, yMax } = this.bounds;
        const yRange = yMax - yMin;
        
        const radialSegments = this.config.radialSegments;
        const samples = this.config.extrusionSamples;
        
        const vertices = [];
        const indices = [];
        const colors = [];
        
        let totalVertexCount = 0;
        
        // Sample along X-axis
        for (let i = 0; i <= samples; i++) {
            const t = i / samples;
            const x = xMin + (xMax - xMin) * t;
            
            // Find nodes at this X position
            const tolerance = (xMax - xMin) / samples * 2.0;
            const nodesAtX = [];
            
            for (let j = 0; j < coords.x.length; j++) {
                if (Math.abs(coords.x[j] - x) < tolerance) {
                    nodesAtX.push({
                        index: j,
                        x: coords.x[j],
                        y: coords.y[j]
                    });
                }
            }
            
            if (nodesAtX.length === 0) continue;
            
            // Sort nodes by Y coordinate
            nodesAtX.sort((a, b) => a.y - b.y);
            
            // Detect separate branches (Y-gaps)
            const branches = this.detectBranches(nodesAtX, yRange);
            
            // Create circular cross-section for each branch
            for (let branchIdx = 0; branchIdx < branches.length; branchIdx++) {
                const branch = branches[branchIdx];
                
                // Calculate center and radius for this branch
                const centerY = (branch.yMin + branch.yMax) / 2;
                const radius = (branch.yMax - branch.yMin) / 2;
                
                // Get average solution value for this branch
                let solutionSum = 0;
                for (let node of branch.nodes) {
                    solutionSum += this.solutionData.values[node.index];
                }
                const avgSolution = solutionSum / branch.nodes.length;
                
                // Normalize for color
                const [min, max] = this.solutionData.range;
                const normalized = (avgSolution - min) / (max - min);
                const color = this.valueToColor(normalized);
                
                // Create circular cross-section
                const startVertexIndex = totalVertexCount;
                
                for (let j = 0; j <= radialSegments; j++) {
                    const theta = (j / radialSegments) * Math.PI * 2;
                    
                    // Circular cross-section in YZ plane
                    const y = centerY + Math.cos(theta) * radius;
                    const z = Math.sin(theta) * radius;
                    
                    vertices.push(x, y, z);
                    colors.push(color.r, color.g, color.b);
                    totalVertexCount++;
                }
                
                // Create triangles connecting to previous ring (if exists)
                if (i > 0 && branchIdx < this.lastBranchCount) {
                    const prevStart = this.lastBranchStarts[branchIdx];
                    const currStart = startVertexIndex;
                    
                    for (let j = 0; j < radialSegments; j++) {
                        const a = prevStart + j;
                        const b = prevStart + j + 1;
                        const c = currStart + j + 1;
                        const d = currStart + j;
                        
                        // Two triangles per quad
                        indices.push(a, b, d);
                        indices.push(b, c, d);
                    }
                }
                
                // Store for next iteration
                if (!this.lastBranchStarts) this.lastBranchStarts = [];
                this.lastBranchStarts[branchIdx] = startVertexIndex;
            }
            
            this.lastBranchCount = branches.length;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(vertices, 3)
        );
        geometry.setAttribute('color', 
            new THREE.Float32BufferAttribute(colors, 3)
        );
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        console.log(`   Tube: ${totalVertexCount} vertices, ${indices.length / 3} triangles`);
        
        // Reset for next creation
        this.lastBranchStarts = null;
        this.lastBranchCount = 0;
        
        return geometry;
    }
    
    /**
     * Detect separate branches by finding Y-gaps
     */
    detectBranches(nodesAtX, yRange) {
        if (nodesAtX.length === 0) return [];
        
        const branches = [];
        let currentBranch = {
            nodes: [nodesAtX[0]],
            yMin: nodesAtX[0].y,
            yMax: nodesAtX[0].y
        };
        
        const gapThreshold = yRange * this.config.branchGapThreshold;
        
        for (let i = 1; i < nodesAtX.length; i++) {
            const prevY = nodesAtX[i - 1].y;
            const currY = nodesAtX[i].y;
            const gap = currY - prevY;
            
            // If gap is too large, start a new branch
            if (gap > gapThreshold) {
                branches.push(currentBranch);
                currentBranch = {
                    nodes: [nodesAtX[i]],
                    yMin: currY,
                    yMax: currY
                };
            } else {
                // Continue current branch
                currentBranch.nodes.push(nodesAtX[i]);
                currentBranch.yMax = currY;
            }
        }
        
        // Add last branch
        branches.push(currentBranch);
        
        return branches;
    }
    
    /**
     * Create Quad-8 geometry for 2D mesh
     */
    createQuad8Geometry() {
        const coords = this.meshData.coordinates;
        const conn = this.meshData.connectivity;
        
        const vertices = [];
        const indices = [];
        let vertexIndex = 0;
        
        for (let elem of conn) {
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(
                    coords.x[nodeId],
                    coords.y[nodeId],
                    0
                );
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
        geometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(vertices, 3)
        );
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        return geometry;
    }
    
    /**
     * Add solution-based colors to geometry
     */
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
    
    /**
     * Color mapping: normalized [0,1] → RGB (viridis-like)
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
        
        return new THREE.Color(r, g, b);
    }
    
    /**
     * Fit mesh to view with proper scaling and centering
     */
    fitMeshToView(mesh) {
        const coords = this.meshData.coordinates;
        const x = coords.x;
        const y = coords.y;

        let minX = x[0], maxX = x[0];
        let minY = y[0], maxY = y[0];

        for (let i = 1; i < x.length; i++) {
            if (x[i] < minX) minX = x[i];
            if (x[i] > maxX) maxX = x[i];
            if (y[i] < minY) minY = y[i];
            if (y[i] > maxY) maxY = y[i];
        }

        const sizeX = maxX - minX;
        const sizeY = maxY - minY;
        const maxDim = Math.max(sizeX, sizeY);
        const targetSize = 50;
        const scale = targetSize / maxDim;

        mesh.scale.set(scale, scale, scale);

        const centerX = (minX + maxX) / 2;
        
        mesh.position.set(
            -centerX * scale,
            -minY * scale,
            0
        );

        console.log(`   Fitted to view: scale=${scale.toFixed(4)}`);
    }
    
    /**
     * Toggle 2D mesh visibility
     */
    set2DMeshVisible(visible) {
        this.config.show2DMesh = visible;
        if (this.mesh2D) this.mesh2D.visible = visible;
    }
    
    /**
     * Toggle 3D extrusion visibility
     */
    set3DExtrusionVisible(visible) {
        this.config.show3DExtrusion = visible;
        if (this.mesh3D) this.mesh3D.visible = visible;
    }
    
    /**
     * Create both 2D and 3D meshes
     */
    createAll() {
        this.create2DMesh();
        this.create3DExtrusion();
    }
    
    /**
     * Clean up resources
     */
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
