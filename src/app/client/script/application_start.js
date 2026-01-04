import { MillimetricScene } from '../script/scene.js';
import { WaveBackground } from '../script/wave-background.js';
import { BirdFlock } from '../script/bird-flock.js';
import { CloudGradient } from '../script/clouds.js';
import { FEMClient } from '../script/fem-client.js';
import { MetricsDisplay } from '../script/metrics-display.js';
import { FEMMeshRendererCPU } from '../script/fem-mesh-renderer-cpu.js';
import { FEMMeshRendererGPU } from '../script/fem-mesh-renderer-gpu.js';
import { MeshExtruderSDF } from '../script/mesh-extruder-sdf.js';
import { MeshExtruderRect } from '../script/mesh-extruder-rect.js';
import { ParticleFlow } from '../script/particle-flow.js';
import { CameraController } from '../script/camera-controller.js';
import { SettingsManager } from './settings.manager.js';
import { initMetricsManager } from './metrics/MetricsManager.js';


// ============================================================================
// Configuration
// ============================================================================
const useGPU = true;               // Use GPU renderer (false = CPU)
const use3DExtrusion = true;       // Enable 3D extrusion of mesh
const useParticleAnimation = true; // Enable particle flow animation

// Extrusion type: 'cylindrical' (SDF tube) or 'rectangular' (standard FEM slab)
const extrusionType = 'rectangular';  // 'cylindrical' | 'rectangular'

// Change these from const to window properties
window.useGPU = true;               
window.use3DExtrusion = true;       
window.useParticleAnimation = true; 
window.extrusionType = 'rectangular'; 
window.autoSwitchToMetrics = true; // New variable for your requested logic

// ============================================================================
// Initialize Three.js scene
// ============================================================================
const container = document.getElementById('three-layer');
const millimetricScene = new MillimetricScene(container);
millimetricScene.setBackgroundEnabled(false);

// ============================================================================
// Initialize FEM client and metrics display
// ============================================================================
const femClient = new FEMClient();
const metricsDisplay = new MetricsDisplay(document.getElementById('hud-metrics'));

// Initialize mesh renderer (for 2D visualization fallback)
const meshRenderer = useGPU 
    ? new FEMMeshRendererGPU(millimetricScene.getScene()) 
    : new FEMMeshRendererCPU(millimetricScene.getScene());

// 3D mesh extruder (initialized when mesh is loaded)
let meshExtruder = null;
let particleFlow = null;
let velocityData = null;

// Queue for incremental updates that arrive before renderer is ready
let pendingIncrements = [];

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

const birdFlock = new BirdFlock("bird-horizon", {
    top: 10,
    height: 50,
    speed: 125,
    count: 8,
    flocking: 0.8,
    distance: 16,
    loop: 0,
    fadeOut: 0.65
});

const clouds = new CloudGradient('#cloud', {
    count: 60,
    colors: ['#ffffff', '#f0fff3', '#f8ffff', '#e8f8f8'],
    blur: [25, 80],
    spread: [10, 25],
    x: [0, 100],
    y: [0, 70]
});
clouds.init();

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

let meshExtruderPromise = null;
let resolveMeshExtruder = null;

function getMeshExtruderPromise() {
    if (meshExtruder) {
        return Promise.resolve(meshExtruder);
    }
    if (!meshExtruderPromise) {
        meshExtruderPromise = new Promise((resolve) => {
            resolveMeshExtruder = resolve;
        });
    }
    return meshExtruderPromise;
}

function resetMeshExtruderPromise() {
    meshExtruderPromise = null;
    resolveMeshExtruder = null;
}

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
    
    // Dispatch event for metrics system
    document.dispatchEvent(new CustomEvent('fem:stageStart', { detail: data }));
});

// Assembly progress - show element count during assembly stage
femClient.on('assembly_progress', (data) => {
    const { elements_done, total_elements } = data;
    metricsDisplay.updateStatus(`${elements_done.toLocaleString()} / ${total_elements.toLocaleString()} elements`);
});

// ============================================================================
// Clear Scene - Called when starting a new solve
// ============================================================================
function clearScene() {

    console.log('Clearing scene for new solve...');

    // Reset meshExtruder promise
    resetMeshExtruderPromise();
    
    // Clear pending increments
    pendingIncrements = [];
    
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

// ============================================================================
// Geometry Worker - runs heavy computation off main thread
// ============================================================================
let geometryWorker = null;
let pendingGeometryCallback = null;

function initGeometryWorker() {
    if (geometryWorker) return;
    
    geometryWorker = new Worker('./script/geometry-worker.js', { type: 'module' });
    
    geometryWorker.onmessage = (e) => {
        const { type, stage, message, result, error } = e.data;
        
        switch (type) {
            case 'progress':
                console.log(`Worker: ${message}`);
                metricsDisplay.updateStatus(message);
                break;
                
            case 'complete':
                console.log('Worker: Geometry computation complete');
                if (pendingGeometryCallback) {
                    pendingGeometryCallback(result);
                    pendingGeometryCallback = null;
                }
                break;
                
            case 'error':
                console.error('Worker error:', error);
                metricsDisplay.updateStatus('Geometry creation failed');
                pendingGeometryCallback = null;
                break;
        }
    };
    
    geometryWorker.onerror = (e) => {
        console.error('Worker error:', e);
        metricsDisplay.updateStatus('Worker error');
    };
}

// ============================================================================
// Mesh Selected - Use Worker for geometry creation (non-blocking)
// ============================================================================
document.addEventListener('meshSelected', (event) => {
    const { mesh, preloadedData, meshLoader } = event.detail;
    
    console.log('Mesh selected:', mesh.name);
    console.log('Preloaded data available:', preloadedData ? 'Yes' : 'No');
    
    // Clear scene and update metrics IMMEDIATELY (lightweight operations)
    clearScene();
    metricsDisplay.reset();
    
    // Update mesh info immediately (from gallery metadata or preloaded data)
    const nodes = preloadedData?.nodes || mesh.nodes;
    const elements = preloadedData?.elements || mesh.elements;
    if (nodes && elements) {
        metricsDisplay.updateMesh(nodes, elements);
    }
    metricsDisplay.updateStatus('Creating geometry...');
    
    // Handle geometry creation asynchronously
    handleGeometryCreation(mesh, preloadedData, meshLoader);
});

async function handleGeometryCreation(mesh, preloadedData, meshLoader) {
    // Get mesh data - either preloaded or wait for it
    let meshData = preloadedData;
    
    if (!meshData && meshLoader) {
        console.log('Waiting for mesh data...');
        try {
            meshData = await meshLoader.load(mesh.file);
        } catch (error) {
            console.error('Failed to load mesh:', error);
            metricsDisplay.updateStatus('Mesh load failed');
            return;
        }
    }
    
    if (!meshData) {
        console.warn('No mesh data available, geometry will be created when mesh_loaded event arrives');
        return;
    }
    
    // Store mesh data for later use
    window.currentMeshData = meshData;
    
    // Create geometry based on mode
    if (use3DExtrusion && extrusionType === 'rectangular') {
        // Use Worker for rectangular extrusion (heavy computation)
        console.log('Starting Worker-based geometry creation...');
        initGeometryWorker();
        
        // Set up callback for when Worker completes
        pendingGeometryCallback = (result) => {
            createThreeJSGeometryFromWorkerResult(result, meshData);
        };
        
        // Send data to Worker
        geometryWorker.postMessage({
            type: 'createGeometry',
            payload: {
                meshData: {
                    coordinates: meshData.coordinates,
                    connectivity: meshData.connectivity,
                    nodes: meshData.nodes,
                    elements: meshData.elements
                },
                config: {
                    zFactor: 1.0
                }
            }
        });
        
    } else if (use3DExtrusion && extrusionType === 'cylindrical') {
        // Cylindrical mode - use existing synchronous approach
        console.log('Creating cylindrical geometry...');
        try {
            meshExtruder = new MeshExtruderSDF(
                millimetricScene.getScene(),
                meshData,
                null,
                {
                    show2DMesh: false,
                    show3DExtrusion: true,
                    tubeOpacity: 0.8
                }
            );
            await meshExtruder.createGeometryOnly();
            
            window.meshExtruder = meshExtruder;
            
            if (cameraController) {
                cameraController.setMeshExtruder(meshExtruder);
            }
            
            millimetricScene.render();
            console.log('3D geometry created (cylindrical)');
            
            if (resolveMeshExtruder) {
                resolveMeshExtruder(meshExtruder);
            }
            
            metricsDisplay.updateStatus('Awaiting solver...');
            
        } catch (error) {
            console.error('Failed to create cylindrical geometry:', error);
            metricsDisplay.updateStatus('Geometry creation failed');
        }
    } else {
        // 2D mode
        console.log('Creating 2D mesh from preloaded data...');
        meshRenderer.loadMesh(meshData);
        millimetricScene.render();
        console.log('2D mesh created from preloaded data');
        metricsDisplay.updateStatus('Awaiting solver...');
    }
}

/**
 * Create Three.js geometry from Worker result
 */
function createThreeJSGeometryFromWorkerResult(result, meshData) {
    console.log('Creating Three.js geometry from Worker result...');
    
    const { bounds, originalBounds, geometry2D, geometry3D, vertexMapping, elementGrid, ySegmentCache } = result;
    
    // Dispose previous meshExtruder if it exists
    if (meshExtruder) {
        meshExtruder.dispose();
        meshExtruder = null;
        window.meshExtruder = null;
    }
    
    // Create MeshExtruderRect with pre-computed data
    meshExtruder = new MeshExtruderRect(
        millimetricScene.getScene(),
        meshData,
        null,  // No solution yet
        {
            show2DMesh: false,
            show3DExtrusion: true,
            zFactor: 1.0,
            extrusionOpacity: 0.8
        }
    );
    
    // Apply pre-computed data from Worker
    meshExtruder.applyWorkerResult({
        bounds,
        originalBounds,
        geometry2D,
        geometry3D,
        vertexMapping,
        elementGrid,
        ySegmentCache
    });
    
    window.meshExtruder = meshExtruder;
    
    // Register with camera controller
    if (cameraController) {
        cameraController.setMeshExtruder(meshExtruder);
    }
    
    millimetricScene.render();
    
    console.log('3D geometry created from Worker result');
    console.log(`   Vertices: ${geometry3D.vertexCount}, Mapping: ${vertexMapping.mapped} mapped, ${vertexMapping.unmapped} unmapped`);
    
    // Resolve promise for any handlers waiting for meshExtruder
    if (resolveMeshExtruder) {
        resolveMeshExtruder(meshExtruder);
    }
    
    metricsDisplay.updateStatus('Awaiting solver...');
}

femClient.on('job_started', (data) => {
    console.log('Job started:', data.job_id);
    // Don't clear scene here - meshSelected already did it
    // Only reset metrics if geometry wasn't preloaded AND no Worker is pending
    if (!meshExtruder && !meshRenderer.meshObject && !pendingGeometryCallback) {
        clearScene();
    }
    metricsDisplay.updateStatus('Running');
    
    // Dispatch event for metrics system
    document.dispatchEvent(new CustomEvent('fem:jobStarted', { detail: data }));
});

// ============================================================================
// Mesh Loaded - Create 3D geometry (fallback if not preloaded)
// ============================================================================
femClient.on('mesh_loaded', async (data) => {
    metricsDisplay.updateMesh(data.nodes, data.elements);
    
    // Dispatch DOM event for metrics system
    document.dispatchEvent(new CustomEvent('fem:meshLoaded', { detail: data }));
    
    // Store mesh data for later use
    window.currentMeshData = data;
    
    // ========================================================================
    // Skip geometry creation if already done from preloaded data
    // ========================================================================
    if (use3DExtrusion && meshExtruder) {
        console.log('3D geometry already created from preloaded data, skipping');
        
        // Apply any queued solution increments
        if (pendingIncrements.length > 0) {
            const latest = pendingIncrements[pendingIncrements.length - 1];
            console.log(`Applying queued 3D increment from iter ${latest.iteration}`);
            meshExtruder.updateSolutionIncremental(latest);
            millimetricScene.render();
            pendingIncrements = [];
        }
        return;
    }
    
    // Skip if Worker is still creating geometry (will be applied when Worker completes)
    if (use3DExtrusion && pendingGeometryCallback) {
        console.log('Worker geometry creation in progress, skipping fallback');
        return;
    }
    
    if (!use3DExtrusion && meshRenderer.meshObject) {
        console.log('2D mesh already created from preloaded data, skipping');
        
        // Apply any queued solution increments
        if (pendingIncrements.length > 0) {
            const latest = pendingIncrements[pendingIncrements.length - 1];
            console.log(`Applying queued 2D increment from iter ${latest.iteration}`);
            meshRenderer.updateSolutionIncremental(latest);
            millimetricScene.render();
            pendingIncrements = [];
        }
        return;
    }
    
    // ========================================================================
    // Create 3D geometry (fallback - only if not preloaded)
    // ========================================================================
    if (use3DExtrusion && data.coordinates && data.connectivity) {
        console.log(`Creating ${extrusionType} geometry (fallback, not preloaded)...`);
        
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

            // Resolve the promise for any waiting handlers
            if (resolveMeshExtruder) {
                resolveMeshExtruder(meshExtruder);
            }
            
            // Apply any queued solution increments
            if (pendingIncrements.length > 0) {
                const latest = pendingIncrements[pendingIncrements.length - 1];
                console.log(`Applying queued 3D increment from iter ${latest.iteration}`);
                meshExtruder.updateSolutionIncremental(latest);
                millimetricScene.render();
                pendingIncrements = [];
            }
            
        } catch (error) {
            console.error('Failed to create early geometry:', error);
        }
    }
    // Standard 2D visualization (if not using 3D extrusion)
    else if (data.coordinates && data.connectivity) {
        meshRenderer.loadMesh(data);
        millimetricScene.render();
        
        // Apply any queued solution increments for 2D mode
        if (pendingIncrements.length > 0) {
            const latest = pendingIncrements[pendingIncrements.length - 1];
            console.log(`Applying queued 2D increment from iter ${latest.iteration}`);
            meshRenderer.updateSolutionIncremental(latest);
            millimetricScene.render();
            pendingIncrements = [];
        }
    }
});

femClient.on('solve_progress', (data) => {
    metricsDisplay.updateProgress(
        data.iteration,
        data.max_iterations,
        data.residual,
        data.etr_seconds
    );
    
    // Dispatch event for metrics system
    document.dispatchEvent(new CustomEvent('fem:solveProgress', { detail: data }));
});

// ============================================================================
// Solution Increment - Update colors incrementally during solve
// ============================================================================
femClient.on('solution_increment', (data) => {
    if (use3DExtrusion) {
        if (meshExtruder) {
            // meshExtruder is ready - apply update directly
            meshExtruder.updateSolutionIncremental(data);
            millimetricScene.render();
            
            console.log(`3D color update: iter ${data.iteration}, ` +
                        `range [${data.chunk_info.min.toFixed(3)}, ${data.chunk_info.max.toFixed(3)}]`);
        } else {
            // Queue for later - keep only most recent
            pendingIncrements = [data];
            console.log(`Queued 3D increment iter ${data.iteration} (meshExtruder not ready)`);
        }
    } else {
        if (meshRenderer.meshObject) {
            // 2D renderer is ready - apply update directly
            meshRenderer.updateSolutionIncremental(data);
            millimetricScene.render();
            
            console.log(`Solution update: iter ${data.iteration}, ` +
                        `range [${data.chunk_info.min.toFixed(3)}, ${data.chunk_info.max.toFixed(3)}]`);
        } else {
            // Queue for later - keep only most recent
            pendingIncrements = [data];
            console.log(`Queued 2D increment iter ${data.iteration} (meshRenderer not ready)`);
        }
    }
});

// ============================================================================
// Solve Complete - Final updates and particle animation
// ============================================================================
femClient.on('solve_complete', async (data) => {
    metricsDisplay.updateStatus('Applying colors...');
    metricsDisplay.updateTotalTime(data.timing_metrics.total_program_time);
    console.log('Solve complete!', data);
    
    // Dispatch event for metrics system (early, so panels can update while colors apply)
    document.dispatchEvent(new CustomEvent('fem:solveComplete', { detail: data }));
    
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
    if (use3DExtrusion) {
        // Wait for meshExtruder to be ready (Promise-based, no timeout)
        console.log('Waiting for meshExtruder...');
        
        try {
            const extruder = await getMeshExtruderPromise();
            console.log('meshExtruder ready');
            
            // Final color update with complete solution
            extruder.updateSolutionColors(solutionData);
            millimetricScene.render();
            
            console.log('Final 3D colors applied');
            
            // ====================================================================
            // Fetch velocity data and create particle animation
            // ====================================================================
            try {
                metricsDisplay.updateStatus('Loading velocity data...');
                
                const velocityUrl = `/solve/${data.job_id}/velocity/binary`;
                const response = await fetch(`${femClient.serverUrl}${femClient.basePath}${velocityUrl}`);
                
                if (response.ok) {
                    const buffer = await response.arrayBuffer();
                    velocityData = parseVelocityBinary(buffer, data.mesh_info.elements);
                    window.velocityData = velocityData;
                    console.log('Velocity data loaded:', velocityData);
                    
                    // Create particle animation
                    if (useParticleAnimation) {
                        metricsDisplay.updateStatus('Building particles...');
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
                                particleMaxLife: 10.0,
                                colorBySpeed: true,
                                extrusionMode: extrusionType,
                            }
                        );
                        
                        window.particleFlow = particleFlow;
                        
                        if (cameraController) {
                            cameraController.setParticleFlow(particleFlow);
                        }
                        
                        particleFlow.start();
                        
                        metricsDisplay.updateStatus('Ready');
                        console.log('Particle animation started');
                    } else {
                        metricsDisplay.updateStatus('Ready');
                    }
                } else {
                    console.warn('Velocity data not available');
                    metricsDisplay.updateStatus('Ready (no velocity)');
                }
            } catch (velocityError) {
                console.warn('Could not fetch velocity:', velocityError);
                metricsDisplay.updateStatus('Ready (velocity error)');
            }
        } catch (err) {
            console.error('Error waiting for meshExtruder:', err);
            metricsDisplay.updateStatus('Error');
        }
    }
    // ========================================================================
    // Standard 2D Mode
    // ========================================================================
    else if (!use3DExtrusion) {
        meshRenderer.updateSolution(solutionData);
        millimetricScene.render();
        metricsDisplay.updateStatus('Ready');
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
    
    let minSpeed = Infinity;
    let maxSpeed = -Infinity;
    
    for (let i = 0; i < count; i++) {
        if (offset + 8 > buffer.byteLength) {
            console.error(`Buffer overflow at element ${i}, offset ${offset}`);
            break;
        }
        
        const vx = view.getFloat32(offset, true);
        offset += 4;
        const vy = view.getFloat32(offset, true);
        offset += 4;
        
        const speed = Math.sqrt(vx * vx + vy * vy);
        
        vel.push([vx, vy]);
        abs_vel.push(speed);
        
        if (speed < minSpeed) minSpeed = speed;
        if (speed > maxSpeed) maxSpeed = speed;
    }
    
    console.log(`Parsed ${vel.length} velocity vectors`);
    console.log(`   Speed range: [${minSpeed.toFixed(3)}, ${maxSpeed.toFixed(3)}]`);
    
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
            const response = await fetch(`${femClient.serverUrl}${femClient.basePath}/solve/${femClient.currentJobId}/velocity/binary`);
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

// **********************
// CONFIGURATION SETTINGS
// **********************
// === UI BINDING EXPOSURES ===
// Toggle particle animation live
window.setParticleAnimation = (enabled) => {
    window.useParticleAnimation = enabled;
    if (enabled) {
        window.startParticles(); // Uses existing helper
    } else {
        window.stopParticles();  // Uses existing helper
    }
};

// Update Particle Density
window.setParticleDensity = (value) => {
    if (window.particleFlow) {
        // Map the 0.1 - 2.0 slider to a base count (e.g., 1000 * value)
        window.particleFlow.updateConfig({ 
            particleCount: Math.floor(1000 * parseFloat(value)) 
        });
    }
};

// Toggle 3D Extrusion live
window.set3DExtrusion = (enabled) => {
    window.use3DExtrusion = enabled;
    window.toggle3DExtrusion(enabled); // Uses existing helper
};

// Background Wave Speed
window.setWaveSpeed = (value) => {
    if (window.waveBackground) {
        window.waveBackground.setSpeed(parseFloat(value));
    }
};

// HUD Opacity (CSS Variable Bridge)
window.setHUDOpacity = (value) => {
    document.documentElement.style.setProperty('--hud-opacity', value);
    // Alternatively, if you want to apply to all .hud panels directly:
    document.querySelectorAll('.hud').forEach(el => {
        el.style.opacity = value;
    });
};

window.settingsManager = new SettingsManager();

// ============================================================================
// Initialize Metrics System
// ============================================================================
const metricsManager = initMetricsManager();
window.metricsManager = metricsManager;
