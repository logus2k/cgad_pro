import { MillimetricScene } from '../script/scene.js';
import { FEMClient } from '../script/fem-client.js';
import { MetricsDisplay } from '../script/metrics-display.js';
import { FEMMeshRendererCPU } from '../script/fem-mesh-renderer-cpu.js';
import { FEMMeshRendererGPU } from '../script/fem-mesh-renderer-gpu.js';
import { MeshExtruderSDF } from '../script/mesh-extruder-sdf.js';
import { MeshExtruderRect } from '../script/mesh-extruder-rect.js';
import { ParticleFlow } from '../script/particle-flow.js';

// ============================================================================
// Configuration
// ============================================================================
const useGPU = false;              // Use GPU renderer (false = CPU)
const use3DExtrusion = true;       // Enable 3D extrusion of mesh
const useParticleAnimation = true; // Enable particle flow animation

// Extrusion type: 'cylindrical' (SDF tube) or 'rectangular' (standard FEM slab)
const extrusionType = 'rectangular';  // 'cylindrical' | 'rectangular'

// ============================================================================
// Initialize Three.js scene
// ============================================================================
const container = document.getElementById('three-layer');
const millimetricScene = new MillimetricScene(container);

// ============================================================================
// Initialize FEM client and metrics display
// ============================================================================
const femClient = new FEMClient('https://logus2k.com/fem');
const metricsDisplay = new MetricsDisplay(document.getElementById('hud-metrics'));

// Initialize mesh renderer (for 2D visualization)
const meshRenderer = useGPU 
    ? new FEMMeshRendererGPU(millimetricScene.getScene()) 
    : new FEMMeshRendererCPU(millimetricScene.getScene());

// 3D mesh extruder (initialized after solve completes)
let meshExtruder = null;
let particleFlow = null;
let velocityData = null;

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

femClient.on('mesh_loaded', (data) => {
    metricsDisplay.updateMesh(data.nodes, data.elements);
    
    // Store mesh data for later use
    window.currentMeshData = data;
    
    // Render mesh if coordinates are available
    if (data.coordinates && data.connectivity) {
        if (!use3DExtrusion) {
            // Standard 2D visualization
            meshRenderer.loadMesh(data);
            millimetricScene.render();
        }
        // If using 3D extrusion, we'll render after solve completes
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

femClient.on('solve_complete', async (data) => {
    metricsDisplay.updateStatus('Complete');
    metricsDisplay.updateTotalTime(data.timing_metrics.total_program_time);
    console.log('Solve complete!', data);
    
    // Check if we have solution data
    if (!data.solution_field) {
        console.error('No solution field in results');
        return;
    }
    
    const solutionData = {
        values: Array.from(data.solution_field),
        range: data.solution_stats.u_range
    };
    
    // ========================================================================
    // 3D Extrusion Mode
    // ========================================================================
    if (use3DExtrusion) {
        try {
            console.log(`Creating 3D ${extrusionType} extrusion...`);
            
            // Fetch velocity data
            velocityData = null;
            try {
                const velocityUrl = `/solve/${data.job_id}/velocity/binary`;
                const response = await fetch(`https://logus2k.com/fem${velocityUrl}`);
                
                if (response.ok) {
                    const buffer = await response.arrayBuffer();
                    velocityData = parseVelocityBinary(buffer, data.mesh_info.elements);
                    console.log('Velocity data loaded:', velocityData);
                } else {
                    console.warn('Velocity data not available, using solution only');
                }
            } catch (velocityError) {
                console.warn('Could not fetch velocity:', velocityError);
            }
            
            // Create mesh extruder based on type
            if (extrusionType === 'cylindrical') {
                // Cylindrical extrusion (SDF-based tube)
                meshExtruder = new MeshExtruderSDF(
                    millimetricScene.getScene(),
                    window.currentMeshData,
                    solutionData,
                    {
                        show2DMesh: false,
                        show3DExtrusion: true,
                        tubeOpacity: 0.8
                    }
                );
            } else {
                // Rectangular extrusion (standard FEM slab)
                meshExtruder = new MeshExtruderRect(
                    millimetricScene.getScene(),
                    window.currentMeshData,
                    solutionData,
                    {
                        show2DMesh: false,
                        show3DExtrusion: true,
                        zFactor: 1.0,
                        extrusionOpacity: 0.8
                    }
                );
            }

            await meshExtruder.createAll();
            millimetricScene.render();

            // Expose to window
            window.meshExtruder = meshExtruder;
            window.velocityData = velocityData;
            
            console.log(`3D ${extrusionType} extrusion created successfully`);
            
            // ================================================================
            // Particle Animation
            // ================================================================
            if (useParticleAnimation && velocityData) {
                console.log('Creating particle flow animation...');
                
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
                    }
                );
                
                window.particleFlow = particleFlow;
                
                // Start animation
                particleFlow.start();
                
                console.log('Particle animation started');
            }
            
        } catch (error) {
            console.error('Failed to create 3D extrusion:', error);
            
            // Fallback to standard 2D visualization
            console.log('Falling back to 2D visualization');
            meshRenderer.loadMesh(window.currentMeshData);
            meshRenderer.updateSolution(solutionData);
            millimetricScene.render();
        }
    } 
    // ========================================================================
    // Standard 2D Mode
    // ========================================================================
    else {
        console.log('Using standard 2D visualization');
        meshRenderer.updateSolution(solutionData);
        millimetricScene.render();
    }
});

femClient.on('solution_increment', (data) => {
    // Only update during solve if NOT using 3D extrusion
    if (!use3DExtrusion) {
        meshRenderer.updateSolutionIncremental(data);
        millimetricScene.render();
        
        console.log(`Solution update: iter ${data.iteration}, ` +
                    `range [${data.chunk_info.min.toFixed(3)}, ${data.chunk_info.max.toFixed(3)}]`);
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
    
    // Read header (number of elements)
    const count = view.getUint32(offset, true);
    offset += 4;
    
    console.log(`Parsing velocity data: ${count} elements (expected: ${expectedElements})`);
    
    if (count !== expectedElements) {
        console.warn(`Element count mismatch: got ${count}, expected ${expectedElements}`);
    }
    
    // Read velocity vectors
    const vel = [];
    const abs_vel = [];
    
    // Calculate expected buffer size
    const expectedSize = 4 + (count * 2 * 4); // header + (count * 2 floats * 4 bytes)
    
    if (buffer.byteLength < expectedSize) {
        throw new Error(`Buffer too small: ${buffer.byteLength} < ${expectedSize}`);
    }
    
    for (let i = 0; i < count; i++) {
        // Check bounds before reading
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

/**
* Toggle 2D mesh visibility
* Usage: toggle2DMesh(false)  // Hide 2D mesh
*/
window.toggle2DMesh = (visible) => {
    if (meshExtruder) {
        meshExtruder.set2DMeshVisible(visible);
        millimetricScene.render();
        console.log(`2D mesh: ${visible ? 'visible' : 'hidden'}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

/**
* Toggle 3D extrusion visibility
* Usage: toggle3DExtrusion(true)  // Show 3D extrusion
*/
window.toggle3DExtrusion = (visible) => {
    if (meshExtruder) {
        meshExtruder.set3DExtrusionVisible(visible);
        millimetricScene.render();
        console.log(`3D extrusion: ${visible ? 'visible' : 'hidden'}`);
    } else {
        console.warn('Mesh extruder not initialized');
    }
};

/**
* Toggle particle animation
* Usage: toggleParticles(true)  // Start particles
*        toggleParticles(false) // Stop particles
*/
window.toggleParticles = (visible) => {
    if (particleFlow) {
        particleFlow.setVisible(visible);
        console.log(`Particles: ${visible ? 'visible' : 'hidden'}`);
    } else if (visible && velocityData && meshExtruder) {
        // Create particle flow if it doesn't exist
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

/**
* Set visualization mode
* Usage: setMode('3d')      // 3D extrusion only
*        setMode('2d')      // 2D mesh only
*        setMode('both')    // Both 2D and 3D
*        setMode('flow')    // 3D extrusion with particle animation
*/
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

/**
* Update particle configuration
* Usage: updateParticles({ particleCount: 2000, speedScale: 0.5 })
*/
window.updateParticles = (config) => {
    if (particleFlow) {
        particleFlow.updateConfig(config);
        console.log('Particle config updated:', config);
    } else {
        console.warn('Particle flow not initialized');
    }
};

/**
* Quick test: Start a solve
* Usage: quickSolve()
*/
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
