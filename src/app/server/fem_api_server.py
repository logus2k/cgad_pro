"""
FastAPI + Socket.IO server for FEM solver.

Usage:
    python fem_api_server.py
    
    Or with uvicorn:
    uvicorn fem_api_server:app --host 0.0.0.0 --port 5867 --reload
"""
import sys
import io
import time
from pathlib import Path

# Add parent directories to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Points to src/
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "cpu"))
sys.path.insert(0, str(PROJECT_ROOT / "gpu"))

# Add static files path for Web client 
CLIENT_DIR = PROJECT_ROOT / "app" / "client"

# Benchmark data directory
BENCHMARK_DIR = PROJECT_ROOT / "app" / "server" / "benchmark"

# Gallery file for model name lookup
GALLERY_FILE = CLIENT_DIR / "gallery_files.json"

import asyncio
import uuid
from typing import Dict, Optional
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import socketio

import struct
import numpy as np

from models import SolverParams, JobStatus, JobResult
from progress_callback import ProgressCallback
from solver_wrapper import SolverWrapper

# Import benchmark service
from benchmark_service import init_benchmark_service, create_benchmark_router


# ============================================================================
# FastAPI Setup
# ============================================================================
app = FastAPI(
    title="FEMulator Pro API",
    description="GPU-Accelerated Finite Element Analysis Server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Benchmark Service Initialization
# ============================================================================
benchmark_service, benchmark_handler = init_benchmark_service(BENCHMARK_DIR, GALLERY_FILE)

# Add benchmark API routes
benchmark_router = create_benchmark_router(benchmark_service)
app.include_router(benchmark_router)

# ============================================================================
# Socket.IO Setup
# ============================================================================
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

# Combine FastAPI + Socket.IO
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
)

# ============================================================================
# Job Management
# ============================================================================
jobs: Dict[str, dict] = {}  # In-memory job storage


# ============================================================================
# Socket.IO Events
# ============================================================================
@sio.event
async def connect(sid, environ):
    """Client connected"""
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Client disconnected"""
    print(f"Client disconnected: {sid}")


@sio.event
async def join_room(sid, data):
    """Client joins a job room to receive updates"""
    job_id = data.get('job_id')
    if job_id:
        await sio.enter_room(sid, job_id)
        await sio.emit('joined', {'job_id': job_id}, room=sid)
        print(f"Client {sid} joined room {job_id}")
        
        # Check if we need to replay missed events for fast jobs
        if job_id in jobs:
            job = jobs[job_id]
            
            # Replay mesh_loaded if mesh was already loaded
            if job.get('mesh_loaded_data'):
                print(f"Replaying mesh_loaded for {job_id}")
                await sio.emit('mesh_loaded', job['mesh_loaded_data'], room=sid)
            
            # Replay solve_complete if job already completed
            if job['status'] == 'completed' and job.get('solve_complete_data'):
                print(f"Replaying solve_complete for {job_id}")
                await sio.emit('solve_complete', job['solve_complete_data'], room=sid)


# ============================================================================
# Background Solver Task
# ============================================================================
async def run_solver_task(job_id: str, params: dict):
    """Run solver in background with Socket.IO callbacks"""
    try:
        jobs[job_id]['status'] = 'running'
        
        loop = asyncio.get_event_loop()
        solver_type = params.get('solver_type', 'cpu')
        callback = ProgressCallback(sio, job_id, solver_type, loop)
        
        # Intercept on_mesh_loaded to store data for replay
        original_on_mesh_loaded = callback.on_mesh_loaded
        def on_mesh_loaded_with_store(nodes, elements, coordinates, connectivity):
            # Store mesh_loaded data for replay if client joins late
            mesh_loaded_data = {
                'job_id': job_id,
                'solver_type': solver_type,
                'nodes': nodes,
                'elements': elements,
                'binary_url': f'/solve/{job_id}/mesh/binary',
                'timestamp': time.time()
            }
            jobs[job_id]['mesh_loaded_data'] = mesh_loaded_data
            jobs[job_id]['mesh_info'] = {'nodes': nodes, 'elements': elements}
            # Call original callback
            original_on_mesh_loaded(nodes, elements, coordinates, connectivity)
        
        callback.on_mesh_loaded = on_mesh_loaded_with_store
        
        wrapper = SolverWrapper(
            solver_type=params['solver_type'],
            params=params,
            progress_callback=callback
        )
        
        results = await loop.run_in_executor(None, wrapper.run)
        
        # Convert CuPy arrays to NumPy before storing
        solution = results['u']
        if hasattr(solution, 'get'):
            solution = solution.get()
        
        velocity = results.get('vel')
        abs_velocity = results.get('abs_vel')
        pressure = results.get('pressure')
        
        if velocity is not None and hasattr(velocity, 'get'):
            velocity = velocity.get()
        if abs_velocity is not None and hasattr(abs_velocity, 'get'):
            abs_velocity = abs_velocity.get()
        if pressure is not None and hasattr(pressure, 'get'):
            pressure = pressure.get()
        
        # Prepare solve_complete data
        solve_complete_data = {
            'job_id': job_id,
            'solver_type': solver_type,
            'converged': results['converged'],
            'iterations': results['iterations'],
            'timing_metrics': results['timing_metrics'],
            'solution_stats': results['solution_stats'],
            'mesh_info': results['mesh_info'],
            'solution_url': f'/solve/{job_id}/solution/binary',
        }
        
        # Update job with results FIRST (before emitting event)
        jobs[job_id].update({
            'status': 'completed',
            'solve_complete_data': solve_complete_data,
            'results': {
                'converged': results['converged'],
                'iterations': results['iterations'],
                'timing_metrics': results['timing_metrics'],
                'solution_stats': results['solution_stats'],
                'mesh_info': results['mesh_info'],
                'u': solution,
                'vel': velocity,
                'abs_vel': abs_velocity,
                'pressure': pressure
            }
        })
        
        # Record benchmark result
        try:
            benchmark_handler.on_solve_complete(
                job_data=jobs[job_id],
                client_config=jobs[job_id].get('client_config')
            )
        except Exception as bench_err:
            print(f"Warning: Benchmark recording failed: {bench_err}")
        
        print(f"Job {job_id} completed and stored successfully\n")
        
        # THEN emit solve_complete event
        await sio.emit('solve_complete', solve_complete_data, room=job_id)
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        solver_type = params.get('solver_type', 'cpu')
        await sio.emit('solve_error', {
            'job_id': job_id,
            'solver_type': solver_type,
            'error': str(e)
        }, room=job_id)
        print(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory after each job to prevent state contamination
        try:
            import cupy as cp
            # Synchronize GPU to ensure all operations are complete
            cp.cuda.Stream.null.synchronize()
            # Clear the memory pool to release all cached allocations
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print(f"GPU memory cleared after job {job_id}")
        except ImportError:
            pass  # CuPy not available (CPU mode)
        except Exception as cleanup_err:
            print(f"Warning: GPU cleanup failed: {cleanup_err}")

# ============================================================================
# REST API Endpoints
# ============================================================================

"""
@app.get("/")
async def root():
    # API health check
    return {
        "status": "ok",
        "service": "FEMulator Pro API",
        "version": "1.0.0"
    }
"""


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        import cupy
        gpu_available = True
    except ImportError:
        gpu_available = False
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "active_jobs": len([j for j in jobs.values() if j['status'] == 'running']),
        "benchmark_records": len(benchmark_service._records)
    }


@app.post("/solve", response_model=JobStatus)
async def start_solve(params: SolverParams):
    """
    Start a new FEM solve job.
    
    Returns job_id for tracking progress via Socket.IO.
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Resolve mesh file path
    mesh_path = Path(params.mesh_file)
    
    # If relative path, resolve against client/mesh directory
    if not mesh_path.is_absolute():
        base_dir = Path(__file__).parent.parent / "client" / "mesh"
        mesh_path = base_dir / mesh_path.name
    
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {mesh_path}")
    
    # Update params with resolved path
    params_dict = params.model_dump()
    params_dict['mesh_file'] = str(mesh_path)
    
    # Extract client_config if provided
    client_config = None
    if params.client_config:
        client_config = params.client_config.model_dump()
    
    # Create job record
    jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'params': params_dict,
        'results': None,
        'error': None,
        'mesh_loaded_data': None,
        'mesh_info': None,
        'solve_complete_data': None,
        'client_config': client_config  # Store client config for benchmark
    }
    
    # Start solver task in background
    asyncio.create_task(run_solver_task(job_id, params_dict))
    
    return JobStatus(
        job_id=job_id,
        status='queued',
        message=f"Job started. Connect to Socket.IO room '{job_id}' for updates."
    )


@app.get("/solve/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get current status of a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job['status'],
        message=job.get('error')
    )


@app.get("/solve/{job_id}/results", response_model=JobResult)
async def get_job_results(job_id: str):
    """Get results of a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Status: {job['status']}"
        )
    
    results = job['results']
    return JobResult(
        job_id=job_id,
        status=job['status'],
        converged=results['converged'],
        iterations=results['iterations'],
        timing_metrics=results['timing_metrics'],
        solution_stats=results['solution_stats'],
        mesh_info=results['mesh_info']
    )


@app.delete("/solve/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Implement actual cancellation (requires threading.Event or similar)
    jobs[job_id]['status'] = 'cancelled'
    
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/meshes")
async def list_meshes():
    """List available mesh files"""
    mesh_dir = Path(__file__).parent.parent.parent / "data" / "input"
    
    if not mesh_dir.exists():
        return {"meshes": []}
    
    meshes = []
    for mesh_file in mesh_dir.glob("*.h5"):
        # Could add metadata parsing here
        meshes.append({
            "name": mesh_file.name,
            "path": str(mesh_file),
            "size_mb": mesh_file.stat().st_size / 1024 / 1024
        })
    
    return {"meshes": meshes}

@app.get("/solve/{job_id}/mesh/binary")
async def get_mesh_binary(job_id: str):
    """Get mesh geometry as binary (efficient transfer)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    mesh_file = job['params']['mesh_file']
    
    # Load mesh from file
    import h5py
    with h5py.File(mesh_file, 'r') as f:
        x = np.array(f['x'], dtype=np.float32)
        y = np.array(f['y'], dtype=np.float32)
        connectivity = np.array(f['quad8'], dtype=np.int32)
    
    # Pack binary format:
    # Header: [num_nodes(4 bytes), num_elements(4 bytes)]
    # Data: [x_array, y_array, connectivity_flat]
    
    buffer = io.BytesIO()
    
    # Write header
    buffer.write(struct.pack('II', len(x), len(connectivity)))
    
    # Write coordinate arrays
    buffer.write(x.tobytes())
    buffer.write(y.tobytes())
    
    # Write connectivity (flatten to 1D)
    buffer.write(connectivity.flatten().tobytes())
    
    # Reset buffer position to beginning
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=mesh_{job_id}.bin",
            "X-Mesh-Format": "quad8-binary-v1"
        }
    )

@app.get("/solve/{job_id}/solution/binary")
async def get_solution_binary(job_id: str):
    """Get final solution field as binary"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed yet. Status: {job['status']}"
        )
    
    # Get solution from results
    results = job.get('results')
    if not results or 'u' not in results:
        raise HTTPException(status_code=404, detail="Solution data not found")
    
    # Extract solution array (might be NumPy or CuPy)
    solution = results['u']
    
    # Handle CuPy arrays
    if hasattr(solution, 'get'):
        # CuPy array - transfer to CPU first
        solution = solution.get()
    
    # Convert to float32 for efficiency
    solution_f32 = np.array(solution, dtype=np.float32)
    
    # Pack solution as binary
    # Format: [num_values(4 bytes), values(float32 array)]
    buffer = io.BytesIO()
    
    # Write header (number of values)
    buffer.write(struct.pack('I', len(solution_f32)))
    
    # Write solution values
    buffer.write(solution_f32.tobytes())
    
    # Reset buffer position
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=solution_{job_id}.bin",
            "X-Solution-Format": "float32-array",
            "X-Solution-Min": str(float(solution_f32.min())),
            "X-Solution-Max": str(float(solution_f32.max()))
        }
    )

@app.get("/solve/{job_id}/velocity/binary")
async def get_velocity_binary(job_id: str):
    """Get velocity field as binary"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results = job.get('results')
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Check if velocity data exists
    if 'vel' not in results:
        raise HTTPException(status_code=404, detail="Velocity data not computed")
    
    # Get velocity array (shape: [Nels, 2])
    vel = results['vel']
    
    # Handle CuPy arrays
    if hasattr(vel, 'get'):
        vel = vel.get()
    
    # Convert to float32
    vel_f32 = np.array(vel, dtype=np.float32)
    
    print(f"Velocity data: shape={vel_f32.shape}, min={vel_f32.min():.3f}, max={vel_f32.max():.3f}")
    
    # Pack as binary: [count(4 bytes), vx0, vy0, vx1, vy1, ...]
    buffer = io.BytesIO()
    
    # Write header (number of elements)
    buffer.write(struct.pack('I', len(vel_f32)))
    
    # Write velocity vectors (already flattened when writing)
    buffer.write(vel_f32.tobytes())
    
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={
            "X-Velocity-Format": "float32-pairs",
            "X-Element-Count": str(len(vel_f32))
        }
    )

# ============================================================================
# Web Client App Setup
# ============================================================================
app.mount(
    "/",
    StaticFiles(directory=CLIENT_DIR, html=True),
    name="frontend"
)

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FEMulator Pro API Server")
    print("="*70)
    print(f"  Starting server on http://localhost:5867")
    print(f"  Socket.IO endpoint: ws://localhost:5867/socket.io")
    print(f"  API docs: http://localhost:5867/docs")
    print(f"  Benchmark API: http://localhost:5867/api/benchmark")
    print("="*70 + "\n")
    
    uvicorn.run(
        "fem_api_server:socket_app",  # Note: must use socket_app, not app
        host="0.0.0.0",
        port=5867,
        reload=True,
        log_level="info"
    )
