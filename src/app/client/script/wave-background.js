/**
 * WaveBackground - Animated wave effect rendered behind the main scene
 * 
 * Usage:
 *   import { WaveBackground } from './wave-background.js';
 *   
 *   const waveBackground = new WaveBackground(renderer, mainScene, mainCamera, {
 *       speed: 2.5,
 *       bottomOffset: 0.35
 *   });
 *   
 *   waveBackground.start(); // Starts animation loop
 */

import * as THREE from '../library/three.module.min.js';

export class WaveBackground {
    constructor(renderer, mainScene, mainCamera, options = {}) {
        this.renderer = renderer;
        this.mainScene = mainScene;
        this.mainCamera = mainCamera;
        
        // Configuration
        this.speed = options.speed ?? 2.5;
        this.bottomOffset = options.bottomOffset ?? 0.35;
        this.segments = options.segments ?? 150;  // Reduced for performance
        this.layerCount = options.layerCount ?? 4;
        this.waterColors = options.waterColors ?? [0xD4EEF8, 0xC5E6F4, 0xB6DEF0, 0xA7D6EC];
        
        // Frame throttling for performance
        this.targetFPS = options.targetFPS ?? 30;
        this.frameInterval = 1000 / this.targetFPS;
        this.lastFrameTime = 0;
        
        // Non-harmonic frequency ratios
        this.PHI = 1.618033988749;
        this.SQRT2 = 1.41421356237;
        this.SQRT3 = 1.73205080757;
        
        // Wave layer config
        this.yOffsets = [0, -0.05, -0.1, -0.15];
        this.baseOpacities = [0.7, 0.5, 0.3, 1.0];
        
        // Create dedicated scene and camera for background
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(
            -window.innerWidth / 200,
            window.innerWidth / 200,
            window.innerHeight / 200,
            -window.innerHeight / 200,
            0.1,
            1000
        );
        this.camera.position.z = 5;
        
        // Wave layers
        this.waveLayers = [];
        
        // Animation state
        this.running = false;
        this.animationId = null;
        
        // Initialize
        this._createWaves();
        this._bindResize();
    }
    
    _createGeometry() {
        const width = (window.innerWidth / 100) * 1.2;
        return new THREE.PlaneGeometry(width, 1, this.segments, 1);
    }
    
    _createWaves() {
        for (let i = 0; i < this.layerCount; i++) {
            const material = new THREE.MeshBasicMaterial({
                color: this.waterColors[i],
                transparent: true,
                opacity: this.baseOpacities[i],
                side: THREE.DoubleSide
            });
            
            const mesh = new THREE.Mesh(this._createGeometry(), material);
            const baseY = -window.innerHeight / 200 + this.bottomOffset + this.yOffsets[i];
            mesh.position.y = baseY;
            mesh.position.x = 0;
            
            this.scene.add(mesh);
            
            // Random starting phases
            const phaseA = Math.random() * Math.PI * 2;
            const phaseB = Math.random() * Math.PI * 2;
            const phaseC = Math.random() * Math.PI * 2;
            
            this.waveLayers.push({
                mesh,
                
                // Wave components
                freqA: 1.2 + Math.random() * 0.8,
                ampA: 0.05 + Math.random() * 0.025,
                speedA: (0.15 + Math.random() * 0.1) * (Math.random() > 0.5 ? 1 : -1),
                phaseA,
                
                freqB: (1.8 + Math.random() * 0.6) * this.PHI,
                ampB: 0.025 + Math.random() * 0.015,
                speedB: (0.1 + Math.random() * 0.075) * this.SQRT2 * (Math.random() > 0.5 ? 1 : -1),
                phaseB,
                
                freqC: (2.5 + Math.random() * 1.0) * this.SQRT3,
                ampC: 0.015 + Math.random() * 0.01,
                speedC: (0.075 + Math.random() * 0.06) * (Math.random() > 0.5 ? 1 : -1),
                phaseC,
                
                freqModSpeed: 0.025 + Math.random() * 0.025,
                freqModAmount: 0.1 + Math.random() * 0.1,
                
                // Opacity oscillation
                baseOpacity: this.baseOpacities[i],
                opacityAmount: 0.25 + Math.random() * 0.65,
                opacitySpeed: (0.2 + Math.random() * 0.3) * (Math.random() > 0.5 ? 1 : -1),
                opacityPhase: Math.random() * Math.PI * 2,
                opacitySpeed2: (0.1 + Math.random() * 0.2) * this.PHI * (Math.random() > 0.5 ? 1 : -1),
                opacityPhase2: Math.random() * Math.PI * 2,
                
                yOffset: this.yOffsets[i]
            });
        }
    }
    
    _updateWaveGeometry(mesh, layer, time) {
        const positions = mesh.geometry.attributes.position.array;
        const freqMod = 1 + layer.freqModAmount * Math.sin(time * layer.freqModSpeed);
        
        for (let j = 0; j <= this.segments; j++) {
            const index = j * 3 + 1;
            const x = positions[j * 3];
            
            const waveA = Math.sin(
                x * layer.freqA * freqMod + time * layer.speedA + layer.phaseA
            ) * layer.ampA;
            
            const waveB = Math.sin(
                x * layer.freqB + time * layer.speedB + layer.phaseB
            ) * layer.ampB;
            
            const waveC = Math.sin(
                x * layer.freqC / freqMod + time * layer.speedC + layer.phaseC
            ) * layer.ampC;
            
            positions[index] = waveA + waveB + waveC;
        }
        
        mesh.geometry.attributes.position.needsUpdate = true;
    }
    
    _updateOpacity(layer, time) {
        const osc1 = Math.sin(time * layer.opacitySpeed + layer.opacityPhase);
        const osc2 = Math.sin(time * layer.opacitySpeed2 + layer.opacityPhase2) * 0.5;
        const combined = (osc1 + osc2) / 1.5;
        
        let newOpacity = layer.baseOpacity + combined * layer.opacityAmount;
        newOpacity = Math.max(0.1, Math.min(1.0, newOpacity));
        
        layer.mesh.material.opacity = newOpacity;
    }
    
    _bindResize() {
        this._onResize = () => {
            const newHalfWidth = window.innerWidth / 200;
            const newHalfHeight = window.innerHeight / 200;
            
            this.camera.left = -newHalfWidth;
            this.camera.right = newHalfWidth;
            this.camera.top = newHalfHeight;
            this.camera.bottom = -newHalfHeight;
            this.camera.updateProjectionMatrix();
            
            const newGeometry = this._createGeometry();
            
            this.waveLayers.forEach(layer => {
                layer.mesh.geometry.dispose();
                layer.mesh.geometry = newGeometry.clone();
                
                const baseY = -newHalfHeight + this.bottomOffset + layer.yOffset;
                layer.mesh.position.y = baseY;
            });
        };
        
        window.addEventListener('resize', this._onResize);
    }
    
    /**
     * Core render logic - renders waves then main scene
     * @param {THREE.Camera} customCamera - Optional custom camera for main scene
     */
    _renderFrame(customCamera) {
        const time = performance.now() * 0.001 * this.speed;
        
        // Update wave geometry and opacity
        this.waveLayers.forEach(layer => {
            this._updateWaveGeometry(layer.mesh, layer, time);
            this._updateOpacity(layer, time);
        });
        
        // Render wave background first (clears canvas)
        this.renderer.autoClear = true;
        this.renderer.render(this.scene, this.camera);
        
        // Clear only depth buffer, then render main scene on top
        this.renderer.autoClear = false;
        this.renderer.clearDepth();
        this.renderer.render(this.mainScene, customCamera || this.mainCamera);
        
        // Restore autoClear for any other renders
        this.renderer.autoClear = true;
    }
    
    /**
     * Animation loop - throttled to target FPS
     */
    _animate() {
        if (!this.running) return;
        
        this.animationId = requestAnimationFrame(() => this._animate());
        
        // Throttle to target FPS
        const now = performance.now();
        const elapsed = now - this.lastFrameTime;
        if (elapsed < this.frameInterval) return;
        this.lastFrameTime = now - (elapsed % this.frameInterval);
        
        this._renderFrame();
    }
    
    /**
     * Render a single frame immediately (for external triggers)
     * @param {THREE.Camera} customCamera - Optional custom camera for main scene
     */
    renderOnce(customCamera) {
        this._renderFrame(customCamera);
    }
    
    /**
     * Start the animation loop
     */
    start() {
        if (this.running) return;
        this.running = true;
        this.lastFrameTime = performance.now();
        this._animate();
    }
    
    /**
     * Stop the animation loop
     */
    stop() {
        this.running = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    /**
     * Set wave animation speed
     * @param {number} speed - Multiplier (1.0 = normal, 2.0 = double speed)
     */
    setSpeed(speed) {
        this.speed = speed;
    }
    
    /**
     * Set target FPS for animation loop
     * @param {number} fps - Target frames per second (default 30)
     */
    setTargetFPS(fps) {
        this.targetFPS = fps;
        this.frameInterval = 1000 / fps;
    }
    
    /**
     * Set vertical offset from bottom
     * @param {number} offset - Offset value (0.5 = default)
     */
    setBottomOffset(offset) {
        this.bottomOffset = offset;
        this._onResize();
    }
    
    /**
     * Set visibility of waves
     * @param {boolean} visible
     */
    setVisible(visible) {
        this.waveLayers.forEach(layer => {
            layer.mesh.visible = visible;
        });
    }
    
    /**
     * Update main camera reference (if camera changes)
     * @param {THREE.Camera} camera
     */
    setMainCamera(camera) {
        this.mainCamera = camera;
    }
    
    /**
     * Dispose all resources
     */
    dispose() {
        this.stop();
        window.removeEventListener('resize', this._onResize);
        
        this.waveLayers.forEach(layer => {
            layer.mesh.geometry.dispose();
            layer.mesh.material.dispose();
            this.scene.remove(layer.mesh);
        });
        
        this.waveLayers = [];
    }
}
