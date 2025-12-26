import * as THREE from '../library/three.module.min.js';

/**
 * FluidVolume - Renders flowing water inside a tube mesh
 * Creates a slightly inset copy of the tube geometry with animated water shader
 * Works alongside (not replacing) the particle system
 */
export class FluidVolume {
    constructor(group, meshExtruder, velocityData, config = {}) {
        this.group = group;
        this.meshExtruder = meshExtruder;
        this.velocityData = velocityData;
        
        this.config = {
            color: new THREE.Color(0x1e90ff),      // Dodger blue
            opacity: 0.6,
            flowSpeed: 0.5,
            insetFactor: 0.95,                      // How much smaller than tube (0.95 = 95% size)
            ...config
        };
        
        this.mesh = null;
        this.clock = new THREE.Clock();
    }
    
    /**
     * Create the fluid volume mesh
     */
    create() {
        this.dispose();
        
        // Get the 3D mesh geometry from the extruder
        const sourceMesh = this.meshExtruder.mesh3D;
        console.log('FluidVolume: sourceMesh =', sourceMesh);
        
        if (!sourceMesh || !sourceMesh.geometry) {
            console.warn('FluidVolume: No 3D mesh available from extruder');
            return;
        }
        
        console.log('FluidVolume: source geometry vertices =', sourceMesh.geometry.attributes.position.count);
        
        // Clone and scale down slightly to fit inside the tube
        const geometry = sourceMesh.geometry.clone();
        const positions = geometry.attributes.position.array;
        const normals = geometry.attributes.normal?.array;
        
        // Inset vertices along normals to make it smaller than the tube
        if (normals) {
            const insetAmount = 0.01;
            for (let i = 0; i < positions.length; i += 3) {
                positions[i] -= normals[i] * insetAmount;
                positions[i + 1] -= normals[i + 1] * insetAmount;
                positions[i + 2] -= normals[i + 2] * insetAmount;
            }
            geometry.attributes.position.needsUpdate = true;
        }
        
        // Water material with animation
        this.uniforms = {
            time: { value: 0 },
            flowSpeed: { value: this.config.flowSpeed }
        };
        
        const material = new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: `
                varying vec3 vWorldPosition;
                
                void main() {
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPos.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float flowSpeed;
                
                varying vec3 vWorldPosition;
                
                // Simple noise
                float hash(vec3 p) {
                    p = fract(p * 0.3183099 + 0.1);
                    p *= 17.0;
                    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
                }
                
                float noise(vec3 p) {
                    vec3 i = floor(p);
                    vec3 f = fract(p);
                    f = f * f * (3.0 - 2.0 * f);
                    return mix(
                        mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
                            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
                        f.z
                    );
                }
                
                void main() {
                    // Flowing noise
                    vec3 p = vWorldPosition * 3.0;
                    p.x -= time * flowSpeed * 5.0;
                    
                    float n = noise(p) + noise(p * 2.0) * 0.5;
                    n = n / 1.5;
                    
                    // Water colors
                    vec3 deepBlue = vec3(0.05, 0.2, 0.4);
                    vec3 lightBlue = vec3(0.2, 0.5, 0.8);
                    vec3 color = mix(deepBlue, lightBlue, n);
                    
                    // Add white caustics/foam
                    float caustic = noise(p * 1.5 - vec3(time * 2.0, 0.0, 0.0));
                    if (caustic > 0.6) {
                        color += vec3(0.3, 0.4, 0.5) * (caustic - 0.6) * 2.0;
                    }
                    
                    gl_FragColor = vec4(color, 0.7);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        
        this.mesh = new THREE.Mesh(geometry, material);
        
        // Copy transform from source mesh
        this.mesh.position.copy(sourceMesh.position);
        this.mesh.scale.copy(sourceMesh.scale);
        this.mesh.rotation.copy(sourceMesh.rotation);
        
        console.log('FluidVolume: mesh position =', this.mesh.position);
        console.log('FluidVolume: mesh scale =', this.mesh.scale);
        console.log('FluidVolume: group =', this.group);
        console.log('FluidVolume: group children before =', this.group.children.length);
        
        this.group.add(this.mesh);
        
        console.log('FluidVolume: group children after =', this.group.children.length);
        console.log('FluidVolume: mesh visible =', this.mesh.visible);
        console.log('FluidVolume: mesh in scene =', this.mesh.parent);
    }
    
    /**
     * Update animation (call each frame)
     */
    update(deltaTime) {
        if (this.uniforms) {
            this.uniforms.time.value += deltaTime || 0.016;
        }
    }
    
    /**
     * Set visibility
     */
    setVisible(visible) {
        if (this.mesh) {
            this.mesh.visible = visible;
        }
    }
    
    /**
     * Check if visible
     */
    isVisible() {
        return this.mesh ? this.mesh.visible : false;
    }
    
    /**
     * Toggle visibility
     */
    toggle() {
        if (this.mesh) {
            this.mesh.visible = !this.mesh.visible;
        }
        return this.isVisible();
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            if (this.mesh.material) this.mesh.material.dispose();
            
            this.mesh = null;
        }
        this.uniforms = null;
    }
}
