import { MillimetricScene } from '../script/scene.js';
import { WaveBackground } from '../script/wave-background.js';
import { FEMClient } from '../script/fem-client.js';
import { MetricsDisplay } from '../script/metrics-display.js';
import { FEMMeshRendererCPU } from '../script/fem-mesh-renderer-cpu.js';
import { FEMMeshRendererGPU } from '../script/fem-mesh-renderer-gpu.js';
import { MeshExtruderSDF } from '../script/mesh-extruder-sdf.js';
import { MeshExtruderRect } from '../script/mesh-extruder-rect.js';
import { ParticleFlow } from '../script/particle-flow.js';
import { CameraController } from '../script/camera-controller.js';

// ============================================================================
// Configuration
// ============================================================================
const useGPU = true;              // Use GPU renderer (false = CPU)
const use3DExtrusion = true;       // Enable 3D extrusion of mesh
const useParticleAnimation = true; // Enable particle flow animation

// Extrusion type: 'cylindrical' (SDF tube) or 'rectangular' (standard FEM slab)
const extrusionType = 'rectangular';  // 'cylindrical' | 'rectangular'

// ============================================================================
// Initialize Three.js scene
// ============================================================================
const container = document.getElementById('three-layer');
const millimetricScene = new MillimetricScene(container);
millimetricScene.setBackgroundEnabled(false);

// ============================================================================
// Initialize FEM client and metrics display
// ============================================================================
const femClient = new FEMClient('https://logus2k.com');
const metricsDisplay = new MetricsDisplay(document.getElementById('hud-metrics'));

// Initialize mesh renderer (for 2D visualization fallback)
const meshRenderer = useGPU 
    ? new FEMMeshRendererGPU(millimetricScene.getScene()) 
    : new FEMMeshRendererCPU(millimetricScene.getScene());

// 3D mesh extruder (initialized when mesh is loaded)
let meshExtruder = null;
let particleFlow = null;
let velocityData = null;

// Camera controller for 2D/3D transitions
let cameraController = null;

// Initialize camera controller
cameraController = new CameraController(
    millimetricScene.getCamera(),
    millimetricScene.getScene(),
    millimetricScene.getRenderer(),
    millimetricScene.getControls(),
    {
        transitionDuration: 1.5,
        margin: 0.1
    }
);

// Set reference to millimetricScene for grid switching
cameraController.setMillimetricScene(millimetricScene);

// Set wave background (pass main scene and camera)
const waveBackground = new WaveBackground(
    millimetricScene.getRenderer(),
    millimetricScene.getScene(),
    millimetricScene.getCamera(),
    {
        speed: 2.5,
        bottomOffset: 0.35
    }
);

// Start the animation loop
waveBackground.start();

// Override millimetricScene.render to always include waves
const originalRender = millimetricScene.render.bind(millimetricScene);
millimetricScene.render = (customCamera) => {
    if (waveBackground && waveBackground.running) {
        // WaveBackground will render both scenes in its loop
        // Just trigger an immediate frame
        waveBackground.renderOnce(customCamera);
    } else {
        originalRender(customCamera);
    }
};

// Expose for console control
window.waveBackground = waveBackground;

window.cameraController = cameraController;

// ============================================================================
// Socket.IO Event Handlers
// ============================================================================

femClient.on('connected', () => {
    metricsDisplay.updateStatus('Connected');
    console.log('Connected to FEM server');
});

femClient.on('stage_start', (data) => {
    metricsDisplay.updateStage(data.stage);
    metricsDisplay.updateStatus('Running');
});

// ============================================================================
// Clear Scene - Called when starting a new solve
// ============================================================================
function clearScene() {
    console.log('Clearing scene for new solve...');
    
    // Clear 2D mesh renderer
    if (meshRenderer) {
        meshRenderer.clear();
    }
    
    // Dispose and clear 3D extruded mesh
    if (meshExtruder) {
        meshExtruder.dispose();
        meshExtruder = null;
        window.meshExtruder = null;
    }
    
    // Stop and dispose particle flow
    if (particleFlow) {
        particleFlow.stop();
        particleFlow.dispose();
        particleFlow = null;
        window.particleFlow = null;
    }
    
    // Clear velocity data
    velocityData = null;
    window.velocityData = null;
    
    // Clear mesh data
    window.currentMeshData = null;
    
    // Clear camera controller references
    if (cameraController) {
        cameraController.setMeshExtruder(null);
        cameraController.setParticleFlow(null);
    }
    
    // Render cleared scene
    millimetricScene.render();
}

window.clearScene = clearScene;

femClient.on('job_started', (data) => {
    console.log('Job started:', data.job_id);
    clearScene();
    metricsDisplay.reset();
});

// ============================================================================
// Mesh Loaded - Create 3D geometry EARLY (before solve)
// ============================================================================
femClient.on('mesh_loaded', async (data) => {
    metricsDisplay.updateMesh(data.nodes, data.elements);
    
    // Store mesh data for later use
    window.currentMeshData = data;
    
    // ========================================================================
    // Create 3D geometry EARLY (before solve starts)
    // ========================================================================
    if (use3DExtrusion && data.coordinates && data.connectivity) {
        console.log(`Creating ${extrusionType} geometry early...`);
        
        try {
            // Dispose existing extruder if any
            if (meshExtruder) {
                meshExtruder.dispose();
                meshExtruder = null;
            }
            
            // Create extruder without solution data (null)
            if (extrusionType === 'cylindrical') {
                meshExtruder = new MeshExtruderSDF(
                    millimetricScene.getScene(),
                    data,
                    null,  // No solution yet
                    {
                        show2DMesh: false,
                        show3DExtrusion: true,
                        tubeOpacity: 0.8
                    }
                );
                await meshExtruder.createGeometryOnly();
            } else {
                meshExtruder = new MeshExtruderRect(
                    millimetricScene.getScene(),
                    data,
                    null,  // No solution yet
                    {
                        show2DMesh: false,
                        show3DExtrusion: true,
                        zFactor: 1.0,
                        extrusionOpacity: 0.8
                    }
                );
                meshExtruder.createGeometryOnly();
            }
            
            window.meshExtruder = meshExtruder;
            
            // Register mesh extruder with camera controller
            if (cameraController) {
                cameraController.setMeshExtruder(meshExtruder);
            }
            
            millimetricScene.render();
            
            console.log('3D geometry created (awaiting solution colors)');
            
        } catch (error) {
            console.error('Failed to create early geometry:', error);
        }
    }
    // Standard 2D visualization (if not using 3D extrusion)
    else if (data.coordinates && data.connectivity) {
        meshRenderer.loadMesh(data);
        millimetricScene.render();
    }
});

femClient.on('solve_progress', (data) => {
    metricsDisplay.updateProgress(
        data.iteration,
        data.max_iterations,
        data.residual,
        data.etr_seconds
    );
});

// ============================================================================
// Solution Increment - Update colors incrementally during solve
// ============================================================================
femClient.on('solution_increment', (data) => {
    if (use3DExtrusion && meshExtruder) {
        // Update 3D extruder colors incrementally
        meshExtruder.updateSolutionIncremental(data);
        millimetricScene.render();
        
        console.log(`3D color update: iter ${data.iteration}, ` +
                    `range [${data.chunk_info.min.toFixed(3)}, ${data.chunk_info.max.toFixed(3)}]`);
    } else if (!use3DExtrusion) {
        // Standard 2D renderer update
        meshRenderer.updateSolutionIncremental(data);
        millimetricScene.render();
        
        console.log(`Solution update: iter ${data.iteration}, ` +
                    `range [${data.chunk_info.min.toFixed(3)}, ${data.chunk_info.max.toFixed(3)}]`);
    }
});

// ============================================================================
// Solve Complete - Final updates and particle animation
// ============================================================================
femClient.on('solve_complete', async (data) => {
    metricsDisplay.updateStatus('Complete');
    metricsDisplay.updateTotalTime(data.timing_metrics.total_program_time);
    console.log('Solve complete!', data);
    
    if (!data.solution_field) {
        console.error('No solution field in results');
        return;
    }
    
    const solutionData = {
        values: Array.from(data.solution_field),
        range: data.solution_stats.u_range
    };
    
    // ========================================================================
    // 3D Extrusion Mode - Final color update
    // ========================================================================
    if (use3DExtrusion && meshExtruder) {
        // Final color update with complete solution
        meshExtruder.updateSolutionColors(solutionData);
        millimetricScene.render();
        
        console.log('Final 3D colors applied');
        
        // ====================================================================
        // Fetch velocity data and create particle animation
        // ====================================================================
        try {
            const velocityUrl = `/solve/${data.job_id}/velocity/binary`;
            const response = await fetch(`https://logus2k.com/fem${velocityUrl}`);
            
            if (response.ok) {
                const buffer = await response.arrayBuffer();
                velocityData = parseVelocityBinary(buffer, data.mesh_info.elements);
                window.velocityData = velocityData;
                console.log('Velocity data loaded:', velocityData);
                
                // Create particle animation (for both cylindrical and rectangular)
                if (useParticleAnimation) {
                    console.log(`Creating particle flow animation (${extrusionType} mode)...`);
                    
                    particleFlow = new ParticleFlow(
                        meshExtruder,
                        velocityData,
                        () => millimetricScene.render(),
                        {
                            particleCount: 1000,
                            speedScale: 0.3,
                            particleSize: 0.02,
                            particleOpacity: 0.9,
                            particleMaxLife: 8.0,
                            colorBySpeed: true,
                            extrusionMode: extrusionType,  // Pass extrusion type
                        }
                    );
                    
                    window.particleFlow = particleFlow;
                    
                    // Register with camera controller for 2D mode
                    if (cameraController) {
                        cameraController.setParticleFlow(particleFlow);
                    }
                    
                    particleFlow.start();
                    
                    console.log('Particle animation started');
                }
            } else {
                console.warn('Velocity data not available');
            }
        } catch (velocityError) {
            console.warn('Could not fetch velocity:', velocityError);
        }
    }
    // ========================================================================
    // Standard 2D Mode
    // ========================================================================
    else if (!use3DExtrusion) {
        meshRenderer.updateSolution(solutionData);
        millimetricScene.render();
    }
});

// ============================================================================
// Helper Functions
// ============================================================================

/**
* Parse binary velocity data
* Format: [count(4 bytes), vx0(4), vy0(4), vx1(4), vy1(4), ...]
*/
function parseVelocityBinary(buffer, expectedElements) {
    const view = new DataView(buffer);
    let offset = 0;
    
    const count = view.getUint32(offset, true);
    offset += 4;
    
    console.log(`Parsing velocity data: ${count} elements (expected: ${expectedElements})`);
    
    if (count !== expectedElements) {
        console.warn(`Element count mismatch: got ${count}, expected ${expectedElements}`);
    }
    
    const vel = [];
    const abs_vel = [];
    
    const expectedSize = 4 + (count * 2 * 4);
    
    if (buffer.byteLength < expectedSize) {
        throw new Error(`Buffer too small: ${buffer.byteLength} < ${expectedSize}`);
    }
    
    for (let i = 0; i < count; i++) {
        if (offset + 8 > buffer.byteLength) {
            console.error(`Buffer overflow at element ${i}, offset ${offset}`);
            break;
        }
        
        const vx = view.getFloat32(offset, true);
        offset += 4;
        const vy = view.getFloat32(offset, true);
        offset += 4;
        
        vel.push([vx, vy]);
        abs_vel.push(Math.sqrt(vx * vx + vy * vy));
    }
    
    console.log(`Parsed ${vel.length} velocity vectors`);
    console.log(`   Speed range: [${Math.min(...abs_vel).toFixed(3)}, ${Math.max(...abs_vel).toFixed(3)}]`);
    
    return { vel, abs_vel };
}

// ============================================================================
// Expose to Window for Console Testing
// ============================================================================

window.femClient = femClient;
window.metricsDisplay = metricsDisplay;
window.meshRenderer = meshRenderer;
window.millimetricScene = millimetricScene;

// ============================================================================
// Console Helper Functions
// ============================================================================

window.toggle2DMesh = (visible) => {
    if (meshExtruder) {
        meshExtruder.set2DMeshVisible(visible);
        millimetricScene.render();
        console.log(`2D mesh: ${visible ? 'visible' : 'hidden'}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

window.toggle3DExtrusion = (visible) => {
    if (meshExtruder) {
        meshExtruder.set3DExtrusionVisible(visible);
        millimetricScene.render();
        console.log(`3D extrusion: ${visible ? 'visible' : 'hidden'}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

window.toggleParticles = (visible) => {
    if (particleFlow) {
        particleFlow.setVisible(visible);
        console.log(`Particles: ${visible ? 'visible' : 'hidden'}`);
    } else if (visible && velocityData && meshExtruder) {
        particleFlow = new ParticleFlow(
            meshExtruder,
            velocityData,
            () => millimetricScene.render(),
            { particleCount: 1000, speedScale: 0.3 }
        );
        window.particleFlow = particleFlow;
        particleFlow.start();
        console.log('Particle flow created and started');
    } else {
        console.warn('Cannot create particles: missing meshExtruder or velocityData');
    }
};

window.setMode = (mode) => {
    if (!meshExtruder) {
        console.warn('Mesh extruder not initialized');
        return;
    }
    
    switch (mode) {
        case '2d':
            meshExtruder.setVisualizationMode('2d');
            if (particleFlow) particleFlow.setVisible(false);
            break;
        case '3d':
            meshExtruder.setVisualizationMode('3d');
            if (particleFlow) particleFlow.setVisible(false);
            break;
        case 'both':
            meshExtruder.setVisualizationMode('both');
            if (particleFlow) particleFlow.setVisible(false);
            break;
        case 'flow':
            meshExtruder.setVisualizationMode('3d');
            toggleParticles(true);
            break;
        default:
            console.warn(`Unknown mode: ${mode}. Use '2d', '3d', 'both', or 'flow'`);
            return;
    }
    
    millimetricScene.render();
    console.log(`Visualization mode: ${mode}`);
};

window.updateParticles = (config) => {
    if (particleFlow) {
        particleFlow.updateConfig(config);
        console.log('Particle config updated:', config);
    } else {
        console.warn('Particle flow not initialized. Use startParticles() first.');
    }
};

/**
 * Start particle animation after mesh is rendered
 * Call this if you rendered without particles initially
 * Usage: startParticles()
 * Options: startParticles({ particleCount: 500, speedScale: 0.5 })
 */
window.startParticles = async (options = {}) => {
    if (!meshExtruder) {
        console.warn('Mesh extruder not initialized. Render a mesh first.');
        return;
    }
    
    if (!velocityData) {
        console.log('Fetching velocity data...');
        try {
            const response = await fetch(`https://logus2k.com/fem/solve/${femClient.currentJobId}/velocity/binary`);
            if (!response.ok) throw new Error('Failed to fetch velocity');
            const buffer = await response.arrayBuffer();
            velocityData = parseVelocityBinary(buffer, meshExtruder.meshData.connectivity.length);
            window.velocityData = velocityData;
            console.log('Velocity data loaded');
        } catch (err) {
            console.error('Could not fetch velocity data:', err);
            return;
        }
    }
    
    if (particleFlow) {
        console.log('Particle flow already exists, restarting...');
        particleFlow.start();
        particleFlow.setVisible(true);
        millimetricScene.render();
        return;
    }
    
    const extrusionType = meshExtruder.constructor.name === 'MeshExtruderSDF' ? 'cylindrical' : 'rectangular';
    
    const config = {
        particleCount: 1000,
        speedScale: 0.3,
        particleSize: 0.02,
        particleOpacity: 0.9,
        particleMaxLife: 8.0,
        colorBySpeed: true,
        extrusionMode: extrusionType,
        ...options
    };
    
    console.log(`Creating particle flow (${extrusionType} mode)...`);
    
    particleFlow = new ParticleFlow(
        meshExtruder,
        velocityData,
        () => millimetricScene.render(),
        config
    );
    
    window.particleFlow = particleFlow;
    
    // Register with camera controller for 2D mode
    if (cameraController) {
        cameraController.setParticleFlow(particleFlow);
    }
    
    particleFlow.start();
    console.log('Particle animation started');
};

/**
 * Stop and remove particles
 * Usage: stopParticles()
 */
window.stopParticles = () => {
    if (particleFlow) {
        particleFlow.stop();
        particleFlow.setVisible(false);
        millimetricScene.render();
        console.log('Particles stopped');
    } else {
        console.warn('No particle flow to stop');
    }
};

/**
 * Toggle grid visibility
 * Usage: toggleGrid() or toggleGrid(true/false)
 */
window.toggleGrid = (visible) => {
    if (millimetricScene) {
        if (visible === undefined) {
            visible = millimetricScene.toggleGrid();
        } else {
            millimetricScene.setGridVisible(visible);
        }
        millimetricScene.render();
        console.log(`Grid: ${visible ? 'visible' : 'hidden'}`);
    }
};

// ============================================================================
// Appearance Controls (ready for UI binding)
// ============================================================================

/**
 * Set mesh brightness
 * @param {number} value - 0.0 (black) to 1.0 (full bright)
 * Usage: setBrightness(0.85)
 * UI: <input type="range" min="0" max="1" step="0.05" oninput="setBrightness(this.value)">
 */
window.setBrightness = (value) => {
    if (meshExtruder) {
        meshExtruder.setBrightness(parseFloat(value));
        millimetricScene.render();
        console.log(`Brightness: ${value}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

/**
 * Set mesh opacity
 * @param {number} value - 0.0 (invisible) to 1.0 (fully opaque)
 * Usage: setOpacity(0.8)
 * UI: <input type="range" min="0" max="1" step="0.05" oninput="setOpacity(this.value)">
 */
window.setOpacity = (value) => {
    if (meshExtruder) {
        meshExtruder.setOpacity(parseFloat(value));
        millimetricScene.render();
        console.log(`Opacity: ${value}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

/**
 * Enable or disable transparency
 * @param {boolean} enabled - true/false
 * Usage: setTransparent(true)
 * UI: <input type="checkbox" onchange="setTransparent(this.checked)">
 */
window.setTransparent = (enabled) => {
    if (meshExtruder) {
        meshExtruder.setTransparent(enabled);
        millimetricScene.render();
        console.log(`Transparent: ${enabled}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

/**
 * Get current appearance settings
 * Usage: getAppearance()
 * Returns: { brightness, opacity, transparent }
 */
window.getAppearance = () => {
    if (meshExtruder) {
        const settings = meshExtruder.getAppearanceSettings();
        console.log('Appearance settings:', settings);
        return settings;
    } else {
        console.warn('Mesh extruder not initialized');
        return null;
    }
};

window.quickSolve = () => {
    femClient.startSolve({
        mesh_file: '/home/logus/env/iscte/cgad_pro/data/input/converted_mesh_v5.h5',
        solver_type: 'cpu',
        max_iterations: 5000,
        progress_interval: 50
    });
};

// ============================================================================
// Log Configuration on Startup
// ============================================================================
console.log('FEMulator Pro Configuration:');
console.log(`   GPU Rendering: ${useGPU ? 'Enabled' : 'Disabled'}`);
console.log(`   3D Extrusion: ${use3DExtrusion ? 'Enabled' : 'Disabled'}`);
console.log(`   Extrusion Type: ${extrusionType}`);
console.log(`   Particle Animation: ${useParticleAnimation ? 'Enabled' : 'Disabled'}`);
console.log('\nConsole helpers available:');
console.log('   toggle2DMesh(true/false)');
console.log('   toggle3DExtrusion(true/false)');
console.log('   toggleParticles(true/false)');
console.log('   setMode("2d" | "3d" | "both" | "flow")');
console.log('   updateParticles({ particleCount, speedScale, ... })');
console.log('   quickSolve()');
console.log('\nAppearance controls (real-time, no regeneration):');
console.log('   setBrightness(0.0 - 1.0)');
console.log('   setOpacity(0.0 - 1.0)');
console.log('   setTransparent(true/false)');
console.log('   getAppearance()');
console.log('\nCamera/View controls:');
console.log('   to2D()          - Animate to 2D view (orthographic, XY plane)');
console.log('   to3D()          - Animate back to 3D view');
console.log('   toggleView()    - Toggle between 2D and 3D');
console.log('   setViewMargin(0.1)      - Set 2D view margin (0-1)');
console.log('   setViewDuration(1.5)    - Set transition duration (seconds)');

// ============================================================================
// Camera/View Console Commands
// ============================================================================

window.to2D = () => {
    if (cameraController) {
        cameraController.transitionTo2D();
    } else {
        console.warn('Camera controller not initialized');
    }
};

window.to3D = () => {
    if (cameraController) {
        cameraController.transitionTo3D();
    } else {
        console.warn('Camera controller not initialized');
    }
};

window.toggleView = () => {
    if (cameraController) {
        cameraController.toggle();
    } else {
        console.warn('Camera controller not initialized');
    }
};

window.setViewMargin = (margin) => {
    if (cameraController) {
        cameraController.setMargin(margin);
        console.log(`View margin set to ${margin}`);
    } else {
        console.warn('Camera controller not initialized');
    }
};

window.setViewDuration = (seconds) => {
    if (cameraController) {
        cameraController.setTransitionDuration(seconds);
        console.log(`Transition duration set to ${seconds}s`);
    } else {
        console.warn('Camera controller not initialized');
    }
};
