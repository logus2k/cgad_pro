import * as THREE from '../library/three.module.min.js';

/**
 * GPU-Accelerated FEM Mesh Renderer
 * Performs Quad-8 shape function interpolation directly on the GPU.
 */
export class FEMMeshRendererGPU {
    constructor(scene) {
        this.scene = scene;
        this.meshObject = null;
        this.meshData = null;
    }

    /**
     * Load mesh using GPU Instancing
     */
    loadMesh(meshData) {
        this.clear();
        this.meshData = meshData;

        // 1. Template Geometry: 6-triangle pattern for a single Quad-8
        const template = new THREE.BufferGeometry();
        const localPositions = new Float32Array([
            -1, -1, 0,  0, -1, 0,  1, -1, 0,  // Nodes 0, 1, 2
             1,  0, 0,  1,  1, 0,             // Nodes 3, 4
             0,  1, 0, -1,  1, 0,             // Nodes 5, 6
            -1,  0, 0                         // Node 7
        ]);
        const indices = [0, 1, 7,  1, 2, 3,  3, 4, 5,  5, 6, 7, 1, 3, 7,  3, 5, 7];
        template.setAttribute('position', new THREE.BufferAttribute(localPositions, 3));
        template.setIndex(indices);

        const instancedGeo = new THREE.InstancedBufferGeometry().copy(template);
        const elementCount = meshData.connectivity.length;
        instancedGeo.instanceCount = elementCount;

        // 2. Position Attributes (8 vec3s)
        for (let i = 0; i < 8; i++) {
            const attrArray = new Float32Array(elementCount * 3);
            for (let e = 0; e < elementCount; e++) {
                const nodeId = meshData.connectivity[e][i];
                attrArray[e * 3 + 0] = meshData.coordinates.x[nodeId];
                attrArray[e * 3 + 1] = meshData.coordinates.y[nodeId];
                attrArray[e * 3 + 2] = 0;
            }
            instancedGeo.setAttribute(`nodeP${i}`, new THREE.InstancedBufferAttribute(attrArray, 3));
        }

        // 3. Packed Solution Buffers (2 vec4s to store 8 node values)
        instancedGeo.setAttribute('nodeV_0_3', new THREE.InstancedBufferAttribute(new Float32Array(elementCount * 4), 4));
        instancedGeo.setAttribute('nodeV_4_7', new THREE.InstancedBufferAttribute(new Float32Array(elementCount * 4), 4));

        const material = new THREE.ShaderMaterial({
            uniforms: { uMin: { value: 0 }, uMax: { value: 1 } },
            vertexShader: this.#getVertexShader(),
            fragmentShader: this.#getFragmentShader(),
            side: THREE.DoubleSide,
            wireframe: false // Default to solid for color visualization
        });

        this.meshObject = new THREE.Mesh(instancedGeo, material);
        this.scene.add(this.meshObject);
        this.fitMeshToView();
    }

    /**
     * Complete solution update (Updates all buffers)
     */
    updateSolution(solutionData) {
        if (!this.meshObject) return;
        const { values, range } = solutionData;
        this.#applyValuesToAttributes(values, range[0], range[1]);
    }

    /**
     * Incremental solution update (Called by solver during iterations)
     */
    updateSolutionIncremental(updateData) {
        if (!this.meshObject) {
            console.warn('updateSolutionIncremental: meshObject not ready');
            return;
        }
        if (!this.meshData) {
            console.warn('updateSolutionIncremental: meshData not ready');
            return;
        }
        console.log('updateSolutionIncremental: applying update');
        // ... rest of method
    }

    /**
     * Internal helper to pack node values into the GPU attributes
     */
    #applyValuesToAttributes(values, min, max) {
        this.meshObject.material.uniforms.uMin.value = min;
        this.meshObject.material.uniforms.uMax.value = max;

        const elementCount = this.meshData.connectivity.length;
        const v03 = this.meshObject.geometry.attributes.nodeV_0_3.array;
        const v47 = this.meshObject.geometry.attributes.nodeV_4_7.array;

        for (let e = 0; e < elementCount; e++) {
            const conn = this.meshData.connectivity[e];
            
            // Unrolling the loop for performance (essential for 200k+ nodes)
            v03[e * 4 + 0] = values[conn[0]] || 0;
            v03[e * 4 + 1] = values[conn[1]] || 0;
            v03[e * 4 + 2] = values[conn[2]] || 0;
            v03[e * 4 + 3] = values[conn[3]] || 0;

            v47[e * 4 + 0] = values[conn[4]] || 0;
            v47[e * 4 + 1] = values[conn[5]] || 0;
            v47[e * 4 + 2] = values[conn[6]] || 0;
            v47[e * 4 + 3] = values[conn[7]] || 0;
        }

        this.meshObject.geometry.attributes.nodeV_0_3.needsUpdate = true;
        this.meshObject.geometry.attributes.nodeV_4_7.needsUpdate = true;
    }

    /**
     * UI Compatibility: Toggle wireframe
     */
    setWireframe(enabled) {
        if (this.meshObject) {
            this.meshObject.material.wireframe = enabled;
        }
    }

    /**
     * UI Compatibility: Stub for color scale logic
     */
    createColorScale() { return null; }

    #getVertexShader() {
        return `
            attribute vec3 nodeP0, nodeP1, nodeP2, nodeP3, nodeP4, nodeP5, nodeP6, nodeP7;
            attribute vec4 nodeV_0_3;
            attribute vec4 nodeV_4_7;
            varying float vValue;

            void main() {
                float xi = position.x;
                float eta = position.y;

                float N[8];
                N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0);
                N[1] = 0.5  * (1.0 - xi * xi) * (1.0 - eta);
                N[2] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0);
                N[3] = 0.5  * (1.0 + xi) * (1.0 - eta * eta);
                N[4] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0);
                N[5] = 0.5  * (1.0 - xi * xi) * (1.0 + eta);
                N[6] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0);
                N[7] = 0.5  * (1.0 - xi) * (1.0 - eta * eta);

                vec3 p = N[0]*nodeP0 + N[1]*nodeP1 + N[2]*nodeP2 + N[3]*nodeP3 + 
                         N[4]*nodeP4 + N[5]*nodeP5 + N[6]*nodeP6 + N[7]*nodeP7;

                vValue = N[0]*nodeV_0_3.x + N[1]*nodeV_0_3.y + N[2]*nodeV_0_3.z + N[3]*nodeV_0_3.w + 
                         N[4]*nodeV_4_7.x + N[5]*nodeV_4_7.y + N[6]*nodeV_4_7.z + N[7]*nodeV_4_7.w;

                gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
            }
        `;
    }

    #getFragmentShader() {
        return `
            uniform float uMin;
            uniform float uMax;
            varying float vValue;

            void main() {
                float t = clamp((vValue - uMin) / (uMax - uMin), 0.0, 1.0);
                vec3 color;
                color.r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
                color.g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
                color.b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
                gl_FragColor = vec4(color, 1.0);
            }
        `;
    }

    fitMeshToView() {
        if (!this.meshObject || !this.meshData) return;

        const coords = this.meshData.coordinates;
        const x = coords.x;
        const y = coords.y;

        // 1. Stack-safe min/max calculation
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

        // 2. Center and "Rest on Floor" Scaling
        this.meshObject.scale.set(scale, scale, scale);
        
        // Calculate the horizontal center
        const centerX = (minX + maxX) / 2;
        
        /**
         * VERTICAL ALIGNMENT LOGIC:
         * Instead of subtracting (minY + maxY) / 2 (which centers it),
         * we subtract only minY. This moves the bottom of the mesh to 0.
         * We then multiply by scale because the position is applied 
         * BEFORE the scale in Three.js's local matrix.
         */
        this.meshObject.position.set(
            -centerX * scale, 
            -minY * scale, 
            0
        );

        // 3. Update Bounding Box for Frustum Culling
        this.meshObject.geometry.boundingBox = new THREE.Box3(
            new THREE.Vector3(minX, minY, -1),
            new THREE.Vector3(maxX, maxY, 1)
        );
        
        console.log(`OK: GPU Mesh Aligned to floor. Scale: ${scale.toFixed(4)}`);
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
