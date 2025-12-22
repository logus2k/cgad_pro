// scene.js

import * as THREE from '../library/three.module.min.js';
import { OrbitControls } from '../library/OrbitControls.js';


export class MillimetricScene {

    constructor(container) {

        this.container = container;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f5f3); // Light theme background
        // this.#setupGradientBackground();

        this.camera = new THREE.PerspectiveCamera(
            25,
            container.clientWidth / container.clientHeight,
            1,
            5000
        );

        // Renderer Setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            powerPreference: "low-power" // Prefers integrated GPU to save energy
        });
        
        // Limit pixel ratio to 2. High-DPI (Retina) screens often 
        // try to render 3x or 4x, which destroys GPU performance.
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        // Controls Setup
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        
        // DISABLING DAMPING: This ensures that when you stop moving the mouse, 
        // the 'change' event stops firing immediately, dropping GPU usage to 0%.
        this.controls.enableDamping = false; 
        
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 1; 
        this.controls.maxDistance = 1000;
        this.controls.maxPolarAngle = Math.PI / 2.0;

        // ---------------------------------------------------------
        // ON-DEMAND RENDERING: No requestAnimationFrame loop.
        // The scene only draws when the camera actually moves.
        // ---------------------------------------------------------
        this.controls.addEventListener('change', () => this.render());

        this.#addLights();
        this.#addMillimetricGrid();
        
        // Initial camera positioning
        this.#fitGridToView(this.gridSize);
        
        // Handle resize (aspect ratio only, no camera repositioning)
        this.#onResize();
        window.addEventListener('resize', () => this.#onResize());
        
        // Initial Draw
        this.render();
    }

    render(customCamera = null) {
        const camera = customCamera || this.camera;
        this.renderer.render(this.scene, camera);
    }

    getScene() {
        return this.scene;
    }
    
    getCamera() {
        return this.camera;
    }
    
    getControls() {
        return this.controls;
    }
    
    getRenderer() {
        return this.renderer;
    }    

    #addLights() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        const dir = new THREE.DirectionalLight(0xffffff, 0.6);
        dir.position.set(10, 20, 10);
        this.scene.add(ambient, dir);
    }

    #addMillimetricGrid() {
        const size = 100;
        
        // Create a group to hold all grids - this allows us to rotate them together
        this.gridGroup = new THREE.Group();
        
        // Grey for the smallest squares (most frequent grid lines)
        const grid1 = new THREE.GridHelper(size, size, 0xd8d8d8, 0xd8d8d8); // Light grey
        grid1.material.opacity = 0.50;
        grid1.material.transparent = true;

        // Soft blue for medium squares
        const grid5 = new THREE.GridHelper(size, size / 5, 0xb3d1ff, 0xb3d1ff); // Light-medium blue
        grid5.material.opacity = 0.80;
        grid5.material.transparent = true;

        // Soft blue for larger squares
        const grid10 = new THREE.GridHelper(size, size / 10, 0x99c2ff, 0x99c2ff); // More defined soft blue
        grid10.material.opacity = 0.90;
        grid10.material.transparent = true;

        this.gridSize = size;
        this.grids = [grid1, grid5, grid10];
        
        // Add grids to group
        this.gridGroup.add(grid1, grid5, grid10);
        this.scene.add(this.gridGroup);
        
        // Store initial (3D) and target (2D) states
        this.grid3DRotation = new THREE.Euler(0, 0, 0);
        this.grid3DPosition = new THREE.Vector3(0, 0, 0);
        this.grid2DRotation = new THREE.Euler(Math.PI / 2, 0, 0);  // Rotated to XY plane
        this.grid2DPosition = new THREE.Vector3(0, 0, -1);  // Slightly behind
        
        // Create a separate large grid for 2D mode (graph paper background)
        this.#create2DBackgroundGrid();
    }
    
    #create2DBackgroundGrid() {
        const size = 1000;  // Large enough to cover any viewport
        
        this.grid2DGroup = new THREE.Group();
        
        // Grey for the smallest squares (1 unit)
        const grid1 = new THREE.GridHelper(size, size, 0xd8d8d8, 0xd8d8d8);
        grid1.material.opacity = 0.50;
        grid1.material.transparent = true;

        // Soft blue for medium squares (5 units)
        const grid5 = new THREE.GridHelper(size, size / 5, 0xb3d1ff, 0xb3d1ff);
        grid5.material.opacity = 0.80;
        grid5.material.transparent = true;

        // Soft blue for larger squares (10 units)
        const grid10 = new THREE.GridHelper(size, size / 10, 0x99c2ff, 0x99c2ff);
        grid10.material.opacity = 0.90;
        grid10.material.transparent = true;

        this.grid2DGroup.add(grid1, grid5, grid10);
        
        // Rotate to XY plane and position behind scene
        this.grid2DGroup.rotation.x = Math.PI / 2;
        this.grid2DGroup.position.z = -1;
        
        // Hidden by default
        this.grid2DGroup.visible = false;
        
        this.scene.add(this.grid2DGroup);
    }
    
    /**
     * Set grid interpolation between 3D and 2D modes
     * @param {number} t - Interpolation factor (0 = 3D, 1 = 2D)
     * @param {number} centerY - Center Y position for 2D mode
     */
    setGridInterpolation(t, centerY = 0) {
        if (!this.gridGroup) return;
        
        // Interpolate 3D grid rotation
        const startRot = this.grid3DRotation;
        const endRot = this.grid2DRotation;
        
        this.gridGroup.rotation.x = startRot.x + (endRot.x - startRot.x) * t;
        this.gridGroup.rotation.y = startRot.y + (endRot.y - startRot.y) * t;
        this.gridGroup.rotation.z = startRot.z + (endRot.z - startRot.z) * t;
        
        // Interpolate 3D grid position (Y moves to centerY in 2D mode)
        const startPos = this.grid3DPosition;
        const endPosY = centerY;
        const endPosZ = this.grid2DPosition.z;
        
        this.gridGroup.position.x = startPos.x;
        this.gridGroup.position.y = startPos.y + (endPosY - startPos.y) * t;
        this.gridGroup.position.z = startPos.z + (endPosZ - startPos.z) * t;
        
        // Fade out 3D grid, fade in 2D background grid
        // 3D grid fades out in first half, 2D grid fades in second half
        if (t < 0.5) {
            this.gridGroup.visible = true;
            if (this.grid2DGroup) this.grid2DGroup.visible = false;
        } else {
            this.gridGroup.visible = false;
            if (this.grid2DGroup) {
                this.grid2DGroup.visible = true;
                this.grid2DGroup.position.y = centerY;
            }
        }
    }
    
    /**
     * Set grid mode directly (for immediate switch without animation)
     */
    setGridMode2D(enabled) {
        this.setGridInterpolation(enabled ? 1 : 0, this.lastCenterY || 0);
    }
    
    /**
     * Store center Y for grid positioning
     */
    set2DGridCenterY(centerY) {
        this.lastCenterY = centerY;
    }

    #fitGridToView(gridSize) {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const aspect = w / h;
        const fovRad = THREE.MathUtils.degToRad(this.camera.fov);
        const halfFovTan = Math.tan(fovRad / 2);

        const distanceV = (gridSize / 2) / halfFovTan;
        const distanceH = (gridSize / 2) / (halfFovTan * aspect);
        const requiredDistance = Math.max(distanceV, distanceH); 
        
        const margin = 0.5;
        const distance = requiredDistance * margin;
        
        this.camera.position.set(distance, distance * 0.9, distance);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        
        this.controls.update();
        this.render(); // Redraw after repositioning
    }

    #setupGradientBackground() {
        // Create canvas for gradient
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const context = canvas.getContext('2d');
        
        // Create vertical gradient
        const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, '#fafafa');     // Very light grey at top
        gradient.addColorStop(1, '#555556ff');   // Soft bluish grey at bottom
        
        context.fillStyle = gradient;
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        // Create texture from canvas
        const texture = new THREE.CanvasTexture(canvas);
        
        // Use it as background
        this.scene.background = texture;
    }

    #onResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);

        // Only fit grid on initial setup, not on every resize
        // Camera position should be preserved
        this.render();
    }
}
