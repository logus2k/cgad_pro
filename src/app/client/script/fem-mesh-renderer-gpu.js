import * as THREE from '../library/three.module.min.js';

export class FEMMeshRendererGPU {
    constructor(scene) {
        this.scene = scene;
        this.meshObject = null;
        this.meshData = null;
    }

    loadMesh(meshData) {
        this.clear();
        this.meshData = meshData;

        // 1. Template Geometry (Remains the same)
        const template = new THREE.BufferGeometry();
        const localPositions = new Float32Array([
            -1, -1, 0,  0, -1, 0,  1, -1, 0,
             1,  0, 0,  1,  1, 0,
             0,  1, 0, -1,  1, 0,
            -1,  0, 0
        ]);
        const indices = [0, 1, 7,  1, 2, 3,  3, 4, 5,  5, 6, 7, 1, 3, 7,  3, 5, 7];
        template.setAttribute('position', new THREE.BufferAttribute(localPositions, 3));
        template.setIndex(indices);

        const instancedGeo = new THREE.InstancedBufferGeometry().copy(template);
        const elementCount = meshData.connectivity.length;
        instancedGeo.instanceCount = elementCount;

        // 2. Position Attributes (8 vec3 = 8 attributes)
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

        // 3. The Packed Solution Buffers (Initialized to zero)
        // We use two vec4 attributes to store 8 values
        instancedGeo.setAttribute('nodeV_0_3', new THREE.InstancedBufferAttribute(new Float32Array(elementCount * 4), 4));
        instancedGeo.setAttribute('nodeV_4_7', new THREE.InstancedBufferAttribute(new Float32Array(elementCount * 4), 4));

        const material = new THREE.ShaderMaterial({
            uniforms: { uMin: { value: 0 }, uMax: { value: 1 } },
            vertexShader: this.#getVertexShader(),
            fragmentShader: this.#getFragmentShader(),
            side: THREE.DoubleSide
        });

        this.meshObject = new THREE.Mesh(instancedGeo, material);
        this.scene.add(this.meshObject);
        this.fitMeshToView();
    }

    /**
     * Update solution values by packing 8 floats into 2 vec4 attributes
     */
    updateSolution(solutionData) {
        if (!this.meshObject) return;
        const { values, range } = solutionData;
        this.meshObject.material.uniforms.uMin.value = range[0];
        this.meshObject.material.uniforms.uMax.value = range[1];

        const elementCount = this.meshData.connectivity.length;
        const v03 = this.meshObject.geometry.attributes.nodeV_0_3.array;
        const v47 = this.meshObject.geometry.attributes.nodeV_4_7.array;

        for (let e = 0; e < elementCount; e++) {
            const conn = this.meshData.connectivity[e];
            // Pack first 4 nodes into nodeV_0_3
            v03[e * 4 + 0] = values[conn[0]] || 0;
            v03[e * 4 + 1] = values[conn[1]] || 0;
            v03[e * 4 + 2] = values[conn[2]] || 0;
            v03[e * 4 + 3] = values[conn[3]] || 0;

            // Pack next 4 nodes into nodeV_4_7
            v47[e * 4 + 0] = values[conn[4]] || 0;
            v47[e * 4 + 1] = values[conn[5]] || 0;
            v47[e * 4 + 2] = values[conn[6]] || 0;
            v47[e * 4 + 3] = values[conn[7]] || 0;
        }

        this.meshObject.geometry.attributes.nodeV_0_3.needsUpdate = true;
        this.meshObject.geometry.attributes.nodeV_4_7.needsUpdate = true;
    }

    #getVertexShader() {
        return `
            attribute vec3 nodeP0, nodeP1, nodeP2, nodeP3, nodeP4, nodeP5, nodeP6, nodeP7;
            attribute vec4 nodeV_0_3; // Contains nodes 0, 1, 2, 3
            attribute vec4 nodeV_4_7; // Contains nodes 4, 5, 6, 7
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

                // Unpack values from the vec4s
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
        if (!this.meshObject) return;
        const box = new THREE.Box3().setFromObject(this.meshObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        this.meshObject.position.sub(center);
        const scale = 50 / Math.max(size.x, size.y);
        this.meshObject.scale.set(scale, scale, scale);
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
