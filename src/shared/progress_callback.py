"""
Progress callback for Socket.IO event emission during FEM solving.
"""
import time
import asyncio
from typing import Optional


class ProgressCallback:
    """Emit events to Socket.IO during solving"""
    
    def __init__(self, socketio, job_id: str):
        self.socketio = socketio
        self.job_id = job_id
    
    def _emit_sync(self, event: str, data: dict):
        """Emit event synchronously by scheduling it in the event loop"""
        try:
            # Try to get the running event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run
                asyncio.create_task(
                    self.socketio.emit(event, data, room=self.job_id)
                )
            else:
                # If no loop is running, run it synchronously
                loop.run_until_complete(
                    self.socketio.emit(event, data, room=self.job_id)
                )
        except RuntimeError:
            # No event loop - create a new one
            asyncio.run(
                self.socketio.emit(event, data, room=self.job_id)
            )
    
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
        """Called after mesh is loaded"""
        self._emit_sync('mesh_loaded', {
            'job_id': self.job_id,
            'nodes': nodes,
            'elements': elements,
            'coordinates': coordinates if nodes < 50000 else None,
            'connectivity': connectivity if elements < 10000 else None,
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
