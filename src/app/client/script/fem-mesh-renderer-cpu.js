// fem-mesh-renderer.js

import * as THREE from '../library/three.module.min.js';

/**
 * FEM Mesh Renderer - Visualizes FEM meshes in Three.js
 */
export class FEMMeshRendererCPU {

    constructor(scene) {
        this.scene = scene;
        this.meshObject = null;
        this.meshData = null;
        
        // Color scale for solution visualization
        this.colorScale = this.createColorScale();
    }
    
    /**
     * Load and display mesh geometry
     */
    loadMesh(meshData) {
        // Remove existing mesh if any
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
        }
        
        this.meshData = meshData;
        
        // Check if we have coordinates
        if (!meshData.coordinates || !meshData.connectivity) {
            console.log('Mesh data received (metadata only):', meshData.nodes, 'nodes');
            return;
        }
        
        console.log('Building mesh geometry with 6-triangle subdivision...');
        
        // Create geometry from Quad-8 elements
        const geometry = this.createGeometryFromQuad8(
            meshData.coordinates,
            meshData.connectivity
        );
        
        // Create material (wireframe initially)
        const material = new THREE.MeshBasicMaterial({
            color: 0x4a90e2,
            wireframe: true,
            side: THREE.DoubleSide
        });
        
        this.meshObject = new THREE.Mesh(geometry, material);
        this.scene.add(this.meshObject);
        
        // Center and scale mesh to fit view
        this.fitMeshToView();
        
        console.log('✅ Mesh rendered:', meshData.nodes, 'nodes');
    }
    
    /**
     * Create Three.js geometry from Quad-8 connectivity using 6 triangles per element
     */
    createGeometryFromQuad8(coordinates, connectivity) {
        const { x, y } = coordinates;
        const geometry = new THREE.BufferGeometry();
        
        // Convert Quad-8 to triangles using all 8 nodes
        const vertices = [];
        const indices = [];
        
        let vertexIndex = 0;
        
        for (let elem of connectivity) {
            // Quad-8: nodes [0,1,2,3,4,5,6,7]
            // 0,2,4,6 are corners; 1,3,5,7 are mid-sides
            
            // Add all 8 vertices for this element
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                vertices.push(
                    x[nodeId],  // x
                    y[nodeId],  // y
                    0           // z (2D mesh on XY plane)
                );
            }
            
            // 6-Triangle Subdivision Pattern (local indices 0-7):
            // This covers the perimeter corners and the interior space
            indices.push(
                // Outer corners to mid-nodes
                vertexIndex + 0, vertexIndex + 1, vertexIndex + 7, // Triangle 1
                vertexIndex + 1, vertexIndex + 2, vertexIndex + 3, // Triangle 2
                vertexIndex + 3, vertexIndex + 4, vertexIndex + 5, // Triangle 3
                vertexIndex + 5, vertexIndex + 6, vertexIndex + 7, // Triangle 4
                
                // Interior bridge triangles
                vertexIndex + 1, vertexIndex + 3, vertexIndex + 7, // Triangle 5
                vertexIndex + 3, vertexIndex + 5, vertexIndex + 7  // Triangle 6
            );
            
            vertexIndex += 8;
        }
        
        // Set geometry attributes
        geometry.setAttribute(
            'position',
            new THREE.Float32BufferAttribute(vertices, 3)
        );
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        
        return geometry;
    }
    
    /**
     * Update mesh colors based on solution values (all 8 nodes)
     */
    updateSolution(solutionData) {
        if (!this.meshObject) {
            console.warn('No mesh loaded yet');
            return;
        }
        
        const { values, range } = solutionData;
        const [min, max] = range;
        
        const positions = this.meshObject.geometry.attributes.position;
        const numVertices = positions.count;
        const colors = new Float32Array(numVertices * 3);
        
        const connectivity = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of connectivity) {
            // Map solution for all 8 nodes per element
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
        
        this.meshObject.geometry.setAttribute(
            'color',
            new THREE.BufferAttribute(colors, 3)
        );
        
        this.meshObject.material.dispose();
        this.meshObject.material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            wireframe: false
        });
    }

    /**
     * Update solution incrementally (all 8 nodes)
     */
    updateSolutionIncremental(updateData) {
        if (!this.meshObject) return;
        
        const { solution_values, chunk_info } = updateData;
        const { min, max } = chunk_info;
        
        const positions = this.meshObject.geometry.attributes.position;
        const numVertices = positions.count;
        
        let colors;
        if (this.meshObject.geometry.attributes.color) {
            colors = this.meshObject.geometry.attributes.color.array;
        } else {
            colors = new Float32Array(numVertices * 3);
            this.meshObject.geometry.setAttribute(
                'color',
                new THREE.BufferAttribute(colors, 3)
            );
        }
        
        const connectivity = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of connectivity) {
            for (let i = 0; i < 8; i++) {
                const nodeId = elem[i];
                const value = solution_values[nodeId] || 0;
                const normalized = (value - min) / (max - min);
                const color = this.valueToColor(normalized);
                
                colors[vertexIndex * 3 + 0] = color.r;
                colors[vertexIndex * 3 + 1] = color.g;
                colors[vertexIndex * 3 + 2] = color.b;
                
                vertexIndex++;
            }
        }
        
        this.meshObject.geometry.attributes.color.needsUpdate = true;
        
        if (this.meshObject.material.wireframe) {
            this.meshObject.material.dispose();
            this.meshObject.material = new THREE.MeshBasicMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                wireframe: false
            });
        }
    }   
    
    /**
     * Map normalized value [0,1] to color (viridis-like colormap)
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
    
    createColorScale() { return null; }
    
    fitMeshToView() {
        if (!this.meshObject) return;
        const box = new THREE.Box3().setFromObject(this.meshObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        this.meshObject.position.sub(center);
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetSize = 50;
        const scale = targetSize / maxDim;
        this.meshObject.scale.set(scale, scale, scale);
    }
    
    setWireframe(enabled) {
        if (this.meshObject) this.meshObject.material.wireframe = enabled;
    }
    
    clear() {
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
            this.meshObject = null;
        }
    }
}
