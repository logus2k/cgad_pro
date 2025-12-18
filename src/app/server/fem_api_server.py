"""
FastAPI + Socket.IO server for FEM solver.

Usage:
    python fem_api_server.py
    
    Or with uvicorn:
    uvicorn fem_api_server:app --host 0.0.0.0 --port 4567 --reload
"""
import sys
import io
from pathlib import Path

# Add parent directories to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Points to src/
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "cpu"))
sys.path.insert(0, str(PROJECT_ROOT / "gpu"))

import asyncio
import uuid
from typing import Dict
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import socketio

import struct
import numpy as np

from models import SolverParams, JobStatus, JobResult
from progress_callback import ProgressCallback
from solver_wrapper import SolverWrapper


# ============================================================================
# Socket.IO Setup
# ============================================================================
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # In production, restrict this!
    logger=True,
    engineio_logger=True
)

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
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Combine FastAPI + Socket.IO
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app
)

# ============================================================================
# Job Management
# ============================================================================
jobs: Dict[str, dict] = {}  # In-memory job storage (use Redis in production)


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
        await sio.enter_room(sid, job_id)  # ← Add await
        await sio.emit('joined', {'job_id': job_id}, room=sid)
        print(f"Client {sid} joined room {job_id}")


# ============================================================================
# Background Solver Task
# ============================================================================
async def run_solver_task(job_id: str, params: dict):
    """Run solver in background with Socket.IO callbacks"""
    try:
        # Update job status
        jobs[job_id]['status'] = 'running'
        
        # Create callback with event loop reference
        loop = asyncio.get_event_loop()
        callback = ProgressCallback(sio, job_id, loop)
        
        # Create and run solver
        wrapper = SolverWrapper(
            solver_type=params['solver_type'],
            params=params,
            progress_callback=callback
        )
        
        # Run in executor to not block event loop
        results = await loop.run_in_executor(None, wrapper.run)
        
        # ← FIX: Convert CuPy arrays to NumPy before storing
        solution = results['u']
        if hasattr(solution, 'get'):
            solution = solution.get()  # Transfer from GPU to CPU
        
        # Update job with results
        jobs[job_id].update({
            'status': 'completed',
            'results': {
                'converged': results['converged'],
                'iterations': results['iterations'],
                'timing_metrics': results['timing_metrics'],
                'solution_stats': results['solution_stats'],
                'mesh_info': results['mesh_info'],
                'u': solution  # ← Store CPU copy
            }
        })
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        await sio.emit('solve_error', {
            'job_id': job_id,
            'error': str(e)
        }, room=job_id)
        print(f"Job {job_id} failed: {e}")

# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "ok",
        "service": "FEMulator Pro API",
        "version": "1.0.0"
    }


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
        "active_jobs": len([j for j in jobs.values() if j['status'] == 'running'])
    }


@app.post("/solve", response_model=JobStatus)
async def start_solve(params: SolverParams):
    """
    Start a new FEM solve job.
    
    Returns job_id for tracking progress via Socket.IO.
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Validate mesh file exists
    mesh_path = Path(params.mesh_file)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail=f"Mesh file not found: {params.mesh_file}")
    
    # Create job record
    jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'params': params.dict(),
        'results': None,
        'error': None
    }
    
    # Start solver task in background
    asyncio.create_task(run_solver_task(job_id, params.dict()))
    
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
    
    # ← FIX: Handle CuPy arrays
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

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FEMulator Pro API Server")
    print("="*70)
    print(f"  Starting server on http://localhost:4567")
    print(f"  Socket.IO endpoint: ws://localhost:4567/socket.io")
    print(f"  API docs: http://localhost:4567/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "fem_api_server:socket_app",  # Note: must use socket_app, not app
        host="0.0.0.0",
        port=4567,
        reload=True,
        log_level="info"
    )
