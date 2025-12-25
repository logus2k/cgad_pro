"""
Progress callback for Socket.IO event emission during FEM solving.
"""
import time
import asyncio
from typing import Optional
import struct


class ProgressCallback:
    """Emit events to Socket.IO during solving"""
    
    def __init__(self, socketio, job_id: str, loop=None):
        self.socketio = socketio
        self.job_id = job_id
        self.loop = loop or asyncio.get_event_loop()
        self.last_solution_update = 0  # Throttle solution updates
    
    def _emit_sync(self, event: str, data: dict):
        """Emit event by scheduling in the async event loop"""
        future = asyncio.run_coroutine_threadsafe(
            self.socketio.emit(event, data, room=self.job_id),
            self.loop
        )
        return future
    
    def on_stage_start(self, stage: str):
        """Called when a stage begins"""
        self._emit_sync('stage_start', {
            'job_id': self.job_id,
            'stage': stage,
            'timestamp': time.time()
        })
    
    def on_stage_complete(self, stage: str, duration: float):
        """Called when a stage completes"""
        self._emit_sync('stage_complete', {
            'job_id': self.job_id,
            'stage': stage,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def on_mesh_loaded(self, nodes: int, elements: int, 
                       coordinates: Optional[dict], connectivity: Optional[list]):
        """Called after mesh is loaded - send metadata only"""
        self._emit_sync('mesh_loaded', {
            'job_id': self.job_id,
            'nodes': nodes,
            'elements': elements,
            # Don't send geometry via Socket.IO - use binary endpoint
            'binary_url': f'/solve/{self.job_id}/mesh/binary',
            'timestamp': time.time()
        })
    
    def on_iteration(self, iteration: int, max_iterations: int,
                     residual: float, relative_residual: float,
                     elapsed_time: float, etr_seconds: float):
        """Called during CG iterations"""
        self._emit_sync('solve_progress', {
            'job_id': self.job_id,
            'stage': 'solving',
            'iteration': iteration,
            'max_iterations': max_iterations,
            'residual': float(residual),
            'relative_residual': float(relative_residual),
            'elapsed_time': elapsed_time,
            'etr_seconds': etr_seconds,
            'progress_percent': 100.0 * iteration / max_iterations,
            'timestamp': time.time()
        })
    
    def on_solution_update(self, iteration: int, solution_chunk: bytes, 
                          chunk_info: dict):
        """
        Send incremental solution update as binary
        
        Args:
            iteration: Current iteration number
            solution_chunk: Binary data (partial solution)
            chunk_info: Metadata about the chunk
        """
        # Throttle updates - only send every 100 iterations
        current_time = time.time()
        if current_time - self.last_solution_update < 1.0:  # Max 1 update/sec
            return
        
        self.last_solution_update = current_time
        
        # Convert binary to base64 for Socket.IO
        import base64
        chunk_b64 = base64.b64encode(solution_chunk).decode('ascii')
        
        self._emit_sync('solution_update', {
            'job_id': self.job_id,
            'iteration': iteration,
            'chunk_data': chunk_b64,
            'chunk_info': chunk_info,
            'timestamp': time.time()
        })
    
    def on_solve_complete(self, converged: bool, iterations: int,
                         timing_metrics: dict, solution_stats: dict,
                         mesh_info: dict):
        """Called when entire solve is complete"""
        self._emit_sync('solve_complete', {
            'job_id': self.job_id,
            'converged': converged,
            'iterations': iterations,
            'timing_metrics': timing_metrics,
            'solution_stats': solution_stats,
            'mesh_info': mesh_info,
            'solution_url': f'/solve/{self.job_id}/solution/binary',
            'timestamp': time.time()
        })

    def on_solution_increment(self, iteration: int, solution):
        """Send incremental solution update as binary (native Socket.IO binary support)"""
        
        # Throttle updates
        current_time = time.time()
        if not hasattr(self, 'last_solution_update'):
            self.last_solution_update = 0
            force_send = True
        else:
            force_send = False
        
        if not force_send and current_time - self.last_solution_update < 0.5:
            return
        
        self.last_solution_update = current_time
        
        # Handle CuPy arrays
        import numpy as np
        if hasattr(solution, 'get'):
            solution = solution.get()
        
        # Convert to float32
        solution_f32 = np.array(solution, dtype=np.float32)
        
        # Compute stats
        sol_min = float(solution_f32.min())
        sol_max = float(solution_f32.max())
        
        # Subsample
        stride = 10
        solution_subsample = solution_f32[::stride]
        
        # Send binary data directly (Socket.IO handles binary natively)
        solution_bytes = solution_subsample.tobytes()
        
        self._emit_sync('solution_increment', {
            'job_id': self.job_id,
            'iteration': iteration,
            'chunk_data': solution_bytes,  # Raw bytes - Socket.IO sends as binary
            'chunk_info': {
                'stride': stride,
                'total_nodes': len(solution_f32),
                'transmitted_nodes': len(solution_subsample),
                'min': sol_min,
                'max': sol_max
            },
            'timestamp': time.time()
        })     
    
    def on_error(self, stage: str, error: str):
        """Called when an error occurs"""
        self._emit_sync('solve_error', {
            'job_id': self.job_id,
            'stage': stage,
            'error': str(error),
            'timestamp': time.time()
        })
