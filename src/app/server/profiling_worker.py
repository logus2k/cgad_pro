"""
Profiling Worker - Background worker for async Nsight profiling.

Runs nsys-wrapped solver executions in a background thread and emits
Socket.IO events for real-time progress updates.

Location: /src/app/server/profiling_worker.py

Usage:
    from profiling_worker import ProfilingWorker
    
    worker = ProfilingWorker(profiling_service, sio)
    worker.start()
    
    # After GPU solve completes:
    worker.enqueue(solver='gpu', mesh_file='mesh.h5', job_id='abc123')
"""

import queue
import threading
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from profiling_service import ProfilingService, ProfileMode, SessionStatus


@dataclass
class ProfilingJob:
    """Profiling job data."""
    solver: str
    mesh_file: str
    job_id: str
    session_id: Optional[str] = None
    enqueued_at: float = 0.0


class ProfilingWorker:
    """
    Background worker for async Nsight profiling.
    
    Processes profiling jobs in a queue, running nsys-wrapped solver
    executions and emitting Socket.IO events for progress updates.
    """
    
    # GPU solver types that support profiling
    GPU_SOLVERS = {'gpu', 'numba_cuda'}
    
    def __init__(
        self,
        profiling_service: ProfilingService,
        socketio,
        event_loop=None
    ):
        """
        Initialize profiling worker.
        
        Args:
            profiling_service: ProfilingService instance
            socketio: AsyncServer instance for emitting events
            event_loop: Event loop for async operations
        """
        self.service = profiling_service
        self.sio = socketio
        self.loop = event_loop
        
        self._queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._current_job: Optional[ProfilingJob] = None
        
        print("[ProfilingWorker] Initialized")
    
    def start(self):
        """Start the worker thread."""
        if self._running:
            print("[ProfilingWorker] Already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ProfilingWorker")
        self._thread.start()
        print("[ProfilingWorker] Started")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        # Put sentinel to unblock queue
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[ProfilingWorker] Stopped")
    
    def enqueue(
        self,
        solver: str,
        mesh_file: str,
        job_id: str
    ) -> Optional[str]:
        """
        Enqueue a profiling job.
        
        Args:
            solver: Solver type (gpu, numba_cuda)
            mesh_file: Path to mesh file
            job_id: Original solve job ID (for linking)
            
        Returns:
            Session ID if queued, None if solver not supported
        """
        # Only profile GPU solvers
        if solver not in self.GPU_SOLVERS:
            print(f"[ProfilingWorker] Skipping non-GPU solver: {solver}")
            return None
        
        # Check if nsys is available
        if not self.service.nsys_available:
            print("[ProfilingWorker] nsys not available, skipping profiling")
            return None
        
        job = ProfilingJob(
            solver=solver,
            mesh_file=mesh_file,
            job_id=job_id,
            enqueued_at=time.time()
        )
        
        self._queue.put(job)
        print(f"[ProfilingWorker] Enqueued job for {solver} / {Path(mesh_file).stem}")
        
        return job_id
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    @property
    def is_busy(self) -> bool:
        """Check if worker is processing a job."""
        return self._current_job is not None
    
    def _run(self):
        """Worker loop - processes queue."""
        print("[ProfilingWorker] Worker loop started")
        
        while self._running:
            try:
                # Wait for job with timeout to allow graceful shutdown
                try:
                    job = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check for sentinel
                if job is None:
                    continue
                
                self._current_job = job
                self._process_job(job)
                self._current_job = None
                
            except Exception as e:
                print(f"[ProfilingWorker] Error in worker loop: {e}")
                import traceback
                traceback.print_exc()
                self._current_job = None
        
        print("[ProfilingWorker] Worker loop ended")
    
    def _process_job(self, job: ProfilingJob):
        """Process a single profiling job."""
        mesh_name = Path(job.mesh_file).stem
        print(f"[ProfilingWorker] Processing: {job.solver} / {mesh_name}")
        
        # Emit queued event
        self._emit('profiling_queued', {
            'session_id': None,  # Will be set after service creates session
            'solver': job.solver,
            'mesh': mesh_name,
            'linked_job_id': job.job_id
        })
        
        try:
            # Start profiled run via service
            # This creates a session and runs nsys in the service's thread
            result = self.service.start_profiled_run(
                solver=job.solver,
                mesh_file=job.mesh_file,
                mode=ProfileMode.TIMELINE.value,
                benchmark_args={
                    'linked_job_id': job.job_id
                }
            )
            
            if 'error' in result:
                self._emit('profiling_failed', {
                    'session_id': None,
                    'solver': job.solver,
                    'mesh': mesh_name,
                    'error': result['error']
                })
                return
            
            session_id = result.get('session_id')
            job.session_id = session_id
            
            # Emit running event
            self._emit('profiling_running', {
                'session_id': session_id,
                'solver': job.solver,
                'mesh': mesh_name,
                'message': 'Executing profiled run...'
            })
            
            # Poll for completion
            self._wait_for_completion(job)
            
        except Exception as e:
            print(f"[ProfilingWorker] Job failed: {e}")
            import traceback
            traceback.print_exc()
            
            self._emit('profiling_failed', {
                'session_id': job.session_id,
                'solver': job.solver,
                'mesh': mesh_name,
                'error': str(e)
            })
    
    def _wait_for_completion(self, job: ProfilingJob):
        """Poll session status until complete."""
        session_id = job.session_id
        mesh_name = Path(job.mesh_file).stem
        max_wait = 600  # 10 minutes max
        poll_interval = 0.5
        elapsed = 0
        last_status = None
        
        while elapsed < max_wait:
            session = self.service.get_session(session_id)
            
            if not session:
                self._emit('profiling_failed', {
                    'session_id': session_id,
                    'error': 'Session not found'
                })
                return
            
            status = session.get('status')
            
            # Emit status change events
            if status != last_status:
                last_status = status
                
                if status == SessionStatus.RUNNING.value:
                    self._emit('profiling_running', {
                        'session_id': session_id,
                        'solver': job.solver,
                        'mesh': mesh_name,
                        'message': 'Executing profiled solver run...'
                    })
                
                elif status == SessionStatus.EXTRACTING.value:
                    self._emit('profiling_extracting', {
                        'session_id': session_id,
                        'solver': job.solver,
                        'mesh': mesh_name,
                        'message': 'Parsing profiling report...'
                    })
                
                elif status == SessionStatus.COMPLETED.value:
                    # Get timeline summary for the complete event
                    timeline_summary = session.get('timeline_summary', {})
                    
                    self._emit('profiling_complete', {
                        'session_id': session_id,
                        'solver': job.solver,
                        'mesh': mesh_name,
                        'linked_job_id': job.job_id,
                        'summary': {
                            'total_events': timeline_summary.get('total_events', 0),
                            'total_duration_ms': timeline_summary.get('total_duration_ms', 0),
                            'categories': timeline_summary.get('categories', {})
                        }
                    })
                    print(f"[ProfilingWorker] Completed: {session_id}")
                    return
                
                elif status == SessionStatus.FAILED.value:
                    self._emit('profiling_failed', {
                        'session_id': session_id,
                        'solver': job.solver,
                        'mesh': mesh_name,
                        'error': session.get('error', 'Unknown error')
                    })
                    return
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout
        self._emit('profiling_failed', {
            'session_id': session_id,
            'solver': job.solver,
            'mesh': mesh_name,
            'error': 'Profiling timeout'
        })
    
    def _emit(self, event: str, data: dict):
        """Emit Socket.IO event."""
        if not self.sio or not self.loop:
            print(f"[ProfilingWorker] Would emit {event}: {data}")
            return
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.sio.emit(event, data),
                self.loop
            )
            # Don't wait for result - fire and forget
        except Exception as e:
            print(f"[ProfilingWorker] Failed to emit {event}: {e}")


# =============================================================================
# Factory function for easy initialization
# =============================================================================

def create_profiling_worker(
    profiling_service: ProfilingService,
    socketio,
    event_loop=None,
    auto_start: bool = True
) -> ProfilingWorker:
    """
    Create and optionally start a profiling worker.
    
    Args:
        profiling_service: ProfilingService instance
        socketio: AsyncServer instance
        event_loop: Event loop (auto-detected if None)
        auto_start: Start worker immediately
        
    Returns:
        ProfilingWorker instance
    """
    if event_loop is None:
        try:
            event_loop = asyncio.get_event_loop()
        except RuntimeError:
            event_loop = asyncio.new_event_loop()
    
    worker = ProfilingWorker(profiling_service, socketio, event_loop)
    
    if auto_start:
        worker.start()
    
    return worker


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Profiling Worker Test ===")
    print("This module should be imported and used with ProfilingService")
    print("See fem_api_server.py for integration example")
