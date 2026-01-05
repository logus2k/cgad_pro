import * as THREE from '../library/three.module.min.js';

/**
 * CameraController - Handles smooth transitions between 3D and 2D views
 * 
 * 3D mode: Perspective camera at user-controlled angle, XZ grid (floor)
 * 2D mode: Orthographic camera looking along -Z axis, XY grid (backdrop)
 */
export class CameraController {

    constructor(perspectiveCamera, scene, renderer, orbitControls, config = {}) {
        this.perspectiveCamera = perspectiveCamera;
        this.scene = scene;
        this.renderer = renderer;
        this.orbitControls = orbitControls;
        
        this.config = {
            transitionDuration: 1.5,      // seconds
            margin: 0.1,                   // 10% margin around mesh in 2D view
            easing: 'easeInOutCubic',
            ...config
        };
        
        // State
        this.is2DMode = false;
        this.axisScale = null;
        this.isTransitioning = false;
        this.transitionProgress = 0;
        this.transitionStartTime = 0;
        
        // Store 3D camera state for return transition
        this.saved3DState = null;
        
        // Create orthographic camera for 2D view
        this.orthoCamera = this.createOrthoCamera();
        
        // Current active camera
        this.activeCamera = this.perspectiveCamera;
        
        // Mesh references (set via setMeshes)
        this.mesh2D = null;
        this.mesh3D = null;
        this.meshExtruder = null;
        
        // Particle flow reference (set via setParticleFlow)
        this.particleFlow = null;
        
        // Scene reference for grid switching
        this.millimetricScene = null;
        
        // Animation state
        this.animationId = null;
        this.startPosition = new THREE.Vector3();
        this.startQuaternion = new THREE.Quaternion();
        this.startUp = new THREE.Vector3();
        this.targetPosition = new THREE.Vector3();
        this.targetQuaternion = new THREE.Quaternion();
        this.targetUp = new THREE.Vector3(0, 1, 0);

        // Default 2D view position (looking at origin from Z axis)
        this.fixed2DPosition = new THREE.Vector3(0, 0, 200);
        this.fixed2DQuaternion = new THREE.Quaternion();
        this.fixed2DUp = new THREE.Vector3(0, 1, 0);
        this.targetCenterY = 0;        
        
        // Callback for render updates
        this.onUpdate = null;
        
        console.log('CameraController initialized');
    }
    
    /**
     * Set reference to MillimetricScene for grid switching
     */
    setMillimetricScene(millimetricScene) {
        this.millimetricScene = millimetricScene;
    }
    
    /**
     * Create orthographic camera sized to renderer
     */
    createOrthoCamera() {
        const aspect = this.renderer.domElement.width / this.renderer.domElement.height;
        const frustumSize = 100;
        
        const camera = new THREE.OrthographicCamera(
            -frustumSize * aspect / 2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            -frustumSize / 2,
            0.1,
            2000
        );
        
        return camera;
    }
    
    /**
     * Update orthographic camera to fit the mesh bounds with margin
     * Also stores the fixed 2D position for animation target
     */
    updateOrthoCameraToFitMesh() {
        if (!this.meshExtruder) return;
        
        const bounds = this.meshExtruder.originalBounds;
        const { xMin, xMax, yMin, yMax } = bounds;
        
        // Get the scale factor applied to the mesh
        const scale = this.meshExtruder.scale || 1;
        
        const width = (xMax - xMin) * scale;
        const height = (yMax - yMin) * scale;
        
        // Mesh center in world space
        const worldCenterX = 0;
        const worldCenterY = height / 2;
        
        // Store for grid animation
        this.targetCenterY = worldCenterY;
        
        // Store in millimetricScene for grid positioning
        if (this.millimetricScene && this.millimetricScene.set2DGridCenterY) {
            this.millimetricScene.set2DGridCenterY(worldCenterY);
        }
        
        // =====================================================================
        // FIXED 2D POSITION - used by both ortho and perspective cameras
        // =====================================================================
        const cameraZ = 200;  // Landing distance for 2D view (higher = more zoomed out)
        this.fixed2DPosition = new THREE.Vector3(worldCenterX, worldCenterY, cameraZ);
        this.fixed2DQuaternion = new THREE.Quaternion();
        this.fixed2DUp = new THREE.Vector3(0, 1, 0);
        
        // =====================================================================
        // Calculate ortho frustum to MATCH what perspective camera sees
        // =====================================================================
        const fovRad = THREE.MathUtils.degToRad(this.perspectiveCamera.fov);
        const perspectiveHeight = 2 * cameraZ * Math.tan(fovRad / 2);
        const perspectiveWidth = perspectiveHeight * this.perspectiveCamera.aspect;
        
        // Use perspective visible area as ortho frustum
        this.orthoCamera.left = -perspectiveWidth / 2;
        this.orthoCamera.right = perspectiveWidth / 2;
        this.orthoCamera.top = perspectiveHeight / 2;
        this.orthoCamera.bottom = -perspectiveHeight / 2;
        this.orthoCamera.updateProjectionMatrix();
        
        // Set ortho camera to fixed position and look at center
        this.orthoCamera.position.copy(this.fixed2DPosition);
        this.orthoCamera.up.copy(this.fixed2DUp);
        this.orthoCamera.lookAt(worldCenterX, worldCenterY, 0);
        this.orthoCamera.updateMatrixWorld();
        
        // Store the quaternion after lookAt
        this.fixed2DQuaternion.copy(this.orthoCamera.quaternion);
        
        console.log(`   Fixed 2D position: (${this.fixed2DPosition.x.toFixed(1)}, ${this.fixed2DPosition.y.toFixed(1)}, ${this.fixed2DPosition.z.toFixed(1)})`);
        console.log(`   Matched frustum: ${perspectiveWidth.toFixed(1)} x ${perspectiveHeight.toFixed(1)}`);
    }
    
    /**
     * Set mesh references for visibility toggling
     */
    setMeshExtruder(meshExtruder) {
        this.meshExtruder = meshExtruder;
        if (meshExtruder) {
            this.mesh2D = meshExtruder.mesh2D;
            this.mesh3D = meshExtruder.mesh3D;
        } else {
            this.mesh2D = null;
            this.mesh3D = null;
        }
    }
    
    /**
     * Set particle flow reference for 2D mode flattening
     */
    setParticleFlow(particleFlow) {
        this.particleFlow = particleFlow;
    }

    /**
     * Set axis scale reference for 2D/3D mode switching
     */
    setAxisScale(axisScale) {
        this.axisScale = axisScale;
    }    
    
    /**
     * Set margin for 2D view (0-1, percentage of mesh size)
     */
    setMargin(margin) {
        this.config.margin = Math.max(0, Math.min(1, margin));
        if (this.is2DMode) {
            this.updateOrthoCameraToFitMesh();
        }
    }
    
    /**
     * Set transition duration in seconds
     */
    setTransitionDuration(seconds) {
        this.config.transitionDuration = Math.max(0.1, seconds);
    }
    
    /**
     * Get the currently active camera
     */
    getActiveCamera() {
        return this.activeCamera;
    }
    
    /**
     * Easing functions
     */
    ease(t) {
        switch (this.config.easing) {
            case 'linear':
                return t;
            case 'easeInOutQuad':
                return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            case 'easeInOutCubic':
            default:
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }
    }
    
    /**
     * Save current 3D camera state
     */
    save3DState() {
        this.saved3DState = {
            position: this.perspectiveCamera.position.clone(),
            quaternion: this.perspectiveCamera.quaternion.clone(),
            up: this.perspectiveCamera.up.clone(),
            target: this.orbitControls ? this.orbitControls.target.clone() : new THREE.Vector3()
        };
    }
    
    /**
     * Transition to 2D view (orthographic, looking down Z-axis)
     */
    transitionTo2D() {
        if (this.is2DMode || this.isTransitioning) {
            console.log('Already in 2D mode or transitioning');
            return;
        }
        
        console.log('Transitioning to 2D view...');

        // Save current 3D state
        this.save3DState();
        
        // Update ortho camera and calculate fixed 2D position
        this.updateOrthoCameraToFitMesh();
        
        // Set up transition parameters - use FIXED 2D position as target
        this.startPosition.copy(this.perspectiveCamera.position);
        this.startQuaternion.copy(this.perspectiveCamera.quaternion);
        this.startUp.copy(this.perspectiveCamera.up);
        
        this.targetPosition.copy(this.fixed2DPosition);
        this.targetQuaternion.copy(this.fixed2DQuaternion);
        this.targetUp.copy(this.fixed2DUp);
        
        // Disable orbit controls during transition
        if (this.orbitControls) {
            this.orbitControls.enabled = false;
        }
        
        // Start transition
        this.isTransitioning = true;
        this.transitionProgress = 0;
        this.transitionStartTime = performance.now();
        this.transitionTarget = '2D';
        
        this.animateTransition();
    }
    
    /**
     * Transition back to 3D view
     */
    transitionTo3D() {
        if (!this.is2DMode || this.isTransitioning) {
            console.log('Already in 3D mode or transitioning');
            return;
        }
        
        if (!this.saved3DState) {
            console.warn('No saved 3D state to return to');
            return;
        }
        
        console.log('Transitioning to 3D view...');
        
        // Set up transition from current ortho position to saved 3D position
        this.startPosition.copy(this.orthoCamera.position);
        this.startQuaternion.copy(this.orthoCamera.quaternion);
        this.startUp.copy(this.orthoCamera.up);
        
        this.targetPosition.copy(this.saved3DState.position);
        this.targetQuaternion.copy(this.saved3DState.quaternion);
        this.targetUp.copy(this.saved3DState.up);
        
        // Start transition
        this.isTransitioning = true;
        this.transitionProgress = 0;
        this.transitionStartTime = performance.now();
        this.transitionTarget = '3D';
        
        this.animateTransition();
    }
    
    /**
     * Toggle between 2D and 3D views
     */
    toggle() {
        if (this.is2DMode) {
            this.transitionTo3D();
        } else {
            this.transitionTo2D();
        }
    }
    
    /**
     * Animation loop for transition
     */
    animateTransition() {
        if (!this.isTransitioning) return;
        
        const elapsed = (performance.now() - this.transitionStartTime) / 1000;
        const duration = this.config.transitionDuration;
        
        this.transitionProgress = Math.min(1, elapsed / duration);
        const easedProgress = this.ease(this.transitionProgress);
        
        // Interpolate position
        const currentPosition = new THREE.Vector3().lerpVectors(
            this.startPosition,
            this.targetPosition,
            easedProgress
        );
        
        // Interpolate rotation (slerp for quaternions)
        const currentQuaternion = new THREE.Quaternion().slerpQuaternions(
            this.startQuaternion,
            this.targetQuaternion,
            easedProgress
        );
        
        // Interpolate up vector
        const currentUp = new THREE.Vector3().lerpVectors(
            this.startUp,
            this.targetUp,
            easedProgress
        );
        
        // Apply to appropriate camera based on progress
        if (this.transitionTarget === '2D') {
            // During transition to 2D, move perspective camera throughout
            this.perspectiveCamera.position.copy(currentPosition);
            this.perspectiveCamera.quaternion.copy(currentQuaternion);
            this.perspectiveCamera.up.copy(currentUp);
            this.activeCamera = this.perspectiveCamera;
            
            // Interpolate grid from 3D (floor) to 2D (backdrop)
            if (this.millimetricScene && this.millimetricScene.setGridInterpolation) {
                this.millimetricScene.setGridInterpolation(easedProgress, this.targetCenterY);
            }
            
            // Gradually flatten particles to Z=0
            if (this.particleFlow) {
                this.particleFlow.setZFlatten(easedProgress);
            }
        } else {
            // During transition to 3D, move perspective camera from ortho position
            this.perspectiveCamera.position.copy(currentPosition);
            this.perspectiveCamera.quaternion.copy(currentQuaternion);
            this.perspectiveCamera.up.copy(currentUp);
            this.activeCamera = this.perspectiveCamera;
            
            // Interpolate grid from 2D (backdrop) back to 3D (floor)
            if (this.millimetricScene && this.millimetricScene.setGridInterpolation) {
                this.millimetricScene.setGridInterpolation(1 - easedProgress, this.targetCenterY);
            }
            
            // Gradually restore particles to full Z
            if (this.particleFlow) {
                this.particleFlow.setZFlatten(1 - easedProgress);
            }
        }
        
        // Update mesh visibility based on progress
        this.updateMeshVisibility(easedProgress);
        
        // Trigger render update
        if (this.onUpdate) {
            this.onUpdate(this.activeCamera);
        }
        
        // Check if transition complete
        if (this.transitionProgress >= 1) {
            this.finishTransition();
            return;
        }
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animateTransition());
    }
    
    /**
     * Update mesh visibility during transition
     */
    updateMeshVisibility(progress) {
        if (!this.meshExtruder) return;
        
        // Get fresh references in case they changed
        this.mesh2D = this.meshExtruder.mesh2D;
        this.mesh3D = this.meshExtruder.mesh3D;
        
        if (this.transitionTarget === '2D') {
            // Transitioning to 2D: keep 3D visible and flatten it
            if (this.mesh3D) {
                this.mesh3D.visible = progress < 0.95;  // Hide just before end
                // Scale Z from 1 to 0 (flatten)
                this.mesh3D.scale.z = this.mesh3D.scale.x * (1 - progress);
            }
            // Show 2D mesh earlier (at 90%) to overlap with flattened 3D
            if (this.mesh2D) {
                this.mesh2D.visible = progress >= 0.9;
            }
        } else {
            // Transitioning to 3D: expand the 3D mesh
            if (this.mesh3D) {
                this.mesh3D.visible = progress > 0.05;  // Show shortly after start
                // Scale Z from 0 to 1 (expand)
                this.mesh3D.scale.z = this.mesh3D.scale.x * progress;
            }
            // Hide 2D mesh after a brief overlap
            if (this.mesh2D) {
                this.mesh2D.visible = progress < 0.1;
            }
        }
    }
    
    /**
     * Complete the transition
     */
    finishTransition() {
        this.isTransitioning = false;
        
        if (this.transitionTarget === '2D') {
            // Debug: compare camera positions at switch moment
            console.log('=== Camera Switch Debug ===');
            console.log('Perspective pos:', this.perspectiveCamera.position.x.toFixed(3), this.perspectiveCamera.position.y.toFixed(3), this.perspectiveCamera.position.z.toFixed(3));
            console.log('Ortho pos:', this.orthoCamera.position.x.toFixed(3), this.orthoCamera.position.y.toFixed(3), this.orthoCamera.position.z.toFixed(3));
            
            // Calculate what perspective camera sees at this distance
            const dist = this.perspectiveCamera.position.z;
            const fovRad = THREE.MathUtils.degToRad(this.perspectiveCamera.fov);
            const perspectiveHeight = 2 * dist * Math.tan(fovRad / 2);
            const perspectiveWidth = perspectiveHeight * this.perspectiveCamera.aspect;
            
            // What ortho camera sees
            const orthoWidth = this.orthoCamera.right - this.orthoCamera.left;
            const orthoHeight = this.orthoCamera.top - this.orthoCamera.bottom;
            
            console.log('Perspective visible area:', perspectiveWidth.toFixed(1), 'x', perspectiveHeight.toFixed(1));
            console.log('Ortho visible area:', orthoWidth.toFixed(1), 'x', orthoHeight.toFixed(1));
            console.log('Size ratio (ortho/persp):', (orthoWidth / perspectiveWidth).toFixed(3));
            
            // Set final mesh state BEFORE camera switch
            if (this.mesh2D) this.mesh2D.visible = true;
            if (this.mesh3D) this.mesh3D.visible = false;
            
            // Flatten particles to Z=0 plane for 2D view
            if (this.particleFlow) {
                this.particleFlow.set2DMode(true);
            }

            if (this.axisScale) {
                this.axisScale.set2DMode(true);
            }            
            
            // Stay with perspective camera to avoid flickering
            // (particles are flattened to Z=0, so perspective distortion is minimal)
            this.activeCamera = this.perspectiveCamera;
            this.is2DMode = true;
            
            // Enable 2D zoom (Z-axis only)
            this.enable2DZoom(true);
            
            console.log('Transition to 2D complete');
        } else {
            // Stay with perspective camera
            this.activeCamera = this.perspectiveCamera;
            this.is2DMode = false;
            
            // Final state already set by updateMeshVisibility, just ensure it's correct
            if (this.mesh2D) this.mesh2D.visible = false;
            if (this.mesh3D) {
                this.mesh3D.visible = true;
                // Restore original Z scale
                this.mesh3D.scale.z = this.mesh3D.scale.x;
            }
            
            // Restore particles to 3D mode
            if (this.particleFlow) {
                this.particleFlow.set2DMode(false);
            }

            if (this.axisScale) {
                this.axisScale.set2DMode(false);
            }            
            
            // Disable 2D zoom
            this.enable2DZoom(false);
            
            // Re-enable orbit controls
            if (this.orbitControls) {
                this.orbitControls.enabled = true;
                if (this.saved3DState) {
                    this.orbitControls.target.copy(this.saved3DState.target);
                }
                this.orbitControls.update();
            }
            
            console.log('Transition to 3D complete');
        }
        
        // Final render with the correct camera
        if (this.onUpdate) {
            this.onUpdate(this.activeCamera);
        }
    }
    
    /**
     * Handle window resize
     */
    onResize(width, height) {
        const aspect = width / height;
        
        // Update perspective camera aspect
        this.perspectiveCamera.aspect = aspect;
        this.perspectiveCamera.updateProjectionMatrix();
        
        // Update orthographic camera aspect (maintain height, adjust width)
        const frustumHeight = this.orthoCamera.top - this.orthoCamera.bottom;
        const frustumWidth = frustumHeight * aspect;
        
        this.orthoCamera.left = -frustumWidth / 2;
        this.orthoCamera.right = frustumWidth / 2;
        this.orthoCamera.updateProjectionMatrix();
        
        // Don't reset camera position - preserve current view
    }
    
    /**
     * Cancel ongoing transition
     */
    cancelTransition() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.isTransitioning = false;
    }
    
    /**
     * Enable/disable 2D zoom (Z-axis only)
     */
    enable2DZoom(enabled) {
        if (enabled) {
            // Create bound handler if not exists
            if (!this._zoom2DHandler) {
                this._zoom2DHandler = this.handle2DZoom.bind(this);
            }
            this.renderer.domElement.addEventListener('wheel', this._zoom2DHandler, { passive: false });
        } else {
            if (this._zoom2DHandler) {
                this.renderer.domElement.removeEventListener('wheel', this._zoom2DHandler);
            }
        }
    }
    
    /**
     * Handle mouse wheel for 2D zoom (Z-axis only)
     */
    handle2DZoom(event) {
        if (!this.is2DMode) return;
        
        event.preventDefault();
        
        const zoomSpeed = 0.1;
        const minZ = 10;   // Minimum camera distance
        const maxZ = 500;  // Maximum camera distance
        
        // Get current Z position
        let z = this.perspectiveCamera.position.z;
        
        // Adjust Z based on scroll direction
        // Scroll up (negative deltaY) = zoom in (decrease Z)
        // Scroll down (positive deltaY) = zoom out (increase Z)
        z += event.deltaY * zoomSpeed;
        
        // Clamp to limits
        z = Math.max(minZ, Math.min(maxZ, z));
        
        // Update camera position (only Z changes)
        this.perspectiveCamera.position.z = z;
        
        // Trigger render
        if (this.onUpdate) {
            this.onUpdate(this.activeCamera);
        }
    }
    
    /**
     * Clean up
     */
    dispose() {
        this.cancelTransition();
        this.enable2DZoom(false);  // Remove event listener
        this.orthoCamera = null;
        this.meshExtruder = null;
        this.mesh2D = null;
        this.mesh3D = null;
    }
}
