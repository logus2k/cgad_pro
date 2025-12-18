// fem-mesh-renderer.js

import * as THREE from '../library/three.module.min.js';


/**
 * FEM Mesh Renderer - Visualizes FEM meshes in Three.js
 */
export class FEMMeshRenderer {

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
        
        console.log('Building mesh geometry...');
        
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
     * Create Three.js geometry from Quad-8 connectivity
     */
    createGeometryFromQuad8(coordinates, connectivity) {
        const { x, y } = coordinates;
        const geometry = new THREE.BufferGeometry();
        
        // Convert Quad-8 to triangles (use only corner nodes: 0,2,4,6)
        const vertices = [];
        const indices = [];
        
        let vertexIndex = 0;
        
        for (let elem of connectivity) {
            // Quad-8: nodes [0,1,2,3,4,5,6,7]
            // Use corners: [0,2,4,6]
            const corners = [elem[0], elem[2], elem[4], elem[6]];
            
            // Add vertices for this quad
            for (let nodeId of corners) {
                vertices.push(
                    x[nodeId],  // x
                    y[nodeId],  // y
                    0           // z (2D mesh on XY plane)
                );
            }
            
            // Create two triangles from quad
            // Triangle 1: 0-1-2
            indices.push(
                vertexIndex + 0,
                vertexIndex + 1,
                vertexIndex + 2
            );
            
            // Triangle 2: 0-2-3
            indices.push(
                vertexIndex + 0,
                vertexIndex + 2,
                vertexIndex + 3
            );
            
            vertexIndex += 4;
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
     * Update mesh colors based on solution values
     */
    updateSolution(solutionData) {
        if (!this.meshObject) {
            console.warn('No mesh loaded yet');
            return;
        }
        
        console.log('Updating solution visualization...');
        
        const { values, range } = solutionData;
        const [min, max] = range;
        
        console.log(`Solution range: [${min.toExponential(3)}, ${max.toExponential(3)}]`);
        
        // Get positions to determine how many vertices we have
        const positions = this.meshObject.geometry.attributes.position;
        const numVertices = positions.count;
        
        // Create color array
        const colors = new Float32Array(numVertices * 3); // RGB for each vertex
        
        // We created 4 vertices per element (quad corners)
        // Need to map solution values (per node) to vertices
        const connectivity = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of connectivity) {
            // Quad-8 corners: [0,2,4,6]
            const corners = [elem[0], elem[2], elem[4], elem[6]];
            
            for (let nodeId of corners) {
                // Get solution value for this node
                const value = values[nodeId];
                const normalized = (value - min) / (max - min);
                const color = this.valueToColor(normalized);
                
                // Set RGB
                colors[vertexIndex * 3 + 0] = color.r;
                colors[vertexIndex * 3 + 1] = color.g;
                colors[vertexIndex * 3 + 2] = color.b;
                
                vertexIndex++;
            }
        }
        
        // Update geometry
        this.meshObject.geometry.setAttribute(
            'color',
            new THREE.BufferAttribute(colors, 3)
        );
        
        // Switch to vertex colors material (solid, not wireframe)
        this.meshObject.material.dispose();
        this.meshObject.material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            wireframe: false  // Solid for color visualization
        });
        
        console.log('✅ Solution visualization updated');
    }

    /**
     * Update solution incrementally during solving
     */
    updateSolutionIncremental(updateData) {
        if (!this.meshObject) {
            console.warn('No mesh loaded yet');
            return;
        }
        
        const { solution_values, chunk_info, iteration } = updateData;
        const { min, max } = chunk_info;
        
        console.log(`Incremental update at iteration ${iteration}, range: [${min.toFixed(3)}, ${max.toFixed(3)}]`);
        
        // Get positions
        const positions = this.meshObject.geometry.attributes.position;
        const numVertices = positions.count;
        
        // Create or get color attribute
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
        
        // Update colors based on solution values
        const connectivity = this.meshData.connectivity;
        
        let vertexIndex = 0;
        for (let elem of connectivity) {
            const corners = [elem[0], elem[2], elem[4], elem[6]];
            
            for (let nodeId of corners) {
                // Get solution value for this node
                const value = solution_values[nodeId] || 0;
                const normalized = (value - min) / (max - min);
                const color = this.valueToColor(normalized);
                
                // Set RGB
                colors[vertexIndex * 3 + 0] = color.r;
                colors[vertexIndex * 3 + 1] = color.g;
                colors[vertexIndex * 3 + 2] = color.b;
                
                vertexIndex++;
            }
        }
        
        // Mark attribute as needing update
        this.meshObject.geometry.attributes.color.needsUpdate = true;
        
        // Switch to colored material if still wireframe
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
        // Simple blue -> cyan -> green -> yellow -> red colormap
        t = Math.max(0, Math.min(1, t)); // Clamp to [0,1]
        
        let r, g, b;
        
        if (t < 0.25) {
            // Blue to Cyan
            const s = t / 0.25;
            r = 0;
            g = s;
            b = 1;
        } else if (t < 0.5) {
            // Cyan to Green
            const s = (t - 0.25) / 0.25;
            r = 0;
            g = 1;
            b = 1 - s;
        } else if (t < 0.75) {
            // Green to Yellow
            const s = (t - 0.5) / 0.25;
            r = s;
            g = 1;
            b = 0;
        } else {
            // Yellow to Red
            const s = (t - 0.75) / 0.25;
            r = 1;
            g = 1 - s;
            b = 0;
        }
        
        return new THREE.Color(r, g, b);
    }
    
    /**
     * Create color scale legend
     */
    createColorScale() {
        // Could create a canvas-based legend here
        return null;
    }
    
    /**
     * Fit mesh to view
     */
    fitMeshToView() {
        if (!this.meshObject) return;
        
        // Compute bounding box
        const box = new THREE.Box3().setFromObject(this.meshObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        // Center mesh
        this.meshObject.position.sub(center);
        
        // Calculate scale to fit in view
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetSize = 50; // Fit to ~50 units
        const scale = targetSize / maxDim;
        
        this.meshObject.scale.set(scale, scale, scale);
        
        console.log(`Mesh centered and scaled: ${maxDim.toFixed(2)} → ${targetSize}`);
    }
    
    /**
     * Toggle wireframe mode
     */
    setWireframe(enabled) {
        if (this.meshObject) {
            this.meshObject.material.wireframe = enabled;
        }
    }
    
    /**
     * Clear mesh from scene
     */
    clear() {
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
            this.meshObject = null;
        }
    }
}
