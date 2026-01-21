"""
Profiling Service - NVIDIA Nsight Systems and Compute integration.

This service:
- Executes benchmark runs wrapped with nsys/ncu profiling
- Extracts metrics from Nsight SQLite exports and CSV outputs
- Provides timeline data suitable for Gantt visualization
- Manages profile session storage and cleanup
- Generates HDF5 binary format for optimized client loading
- Emits events via callback for real-time Socket.IO updates

Location: /src/app/server/profiling_service.py
Data directory: /data/profiles/

Dependencies:
- NVIDIA Nsight Systems (nsys) in PATH
- NVIDIA Nsight Compute (ncu) in PATH
- sqlite3 (standard library)
- h5py (for HDF5 export)
"""

import json
import sqlite3
import subprocess
import csv
import shutil
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# HDF5 transformer for GPU-accelerated nsys HDF5 to client HDF5 conversion
try:
    from profiling_hdf5_transformer import ProfilingHDF5Transformer
except ImportError:
    try:
        from .profiling_hdf5_transformer import ProfilingHDF5Transformer
    except ImportError:
        ProfilingHDF5Transformer = None
        print("[Profiling] Warning: profiling_hdf5_transformer not found, HDF5 transform disabled")


# =============================================================================
# Constants and Enums
# =============================================================================

class ProfileMode(str, Enum):
    """Profiling mode selection."""
    TIMELINE = "timeline"    # nsys only - fast, low overhead
    KERNELS = "kernels"      # ncu only - slow, detailed kernel metrics
    FULL = "full"            # nsys + ncu sequentially


class SessionStatus(str, Enum):
    """Profile session status."""
    PENDING = "pending"
    RUNNING = "running"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


# NVTX color mapping for consistent visualization
NVTX_COLORS = {
    "load_mesh": "#3498db",        # Blue
    "assemble_system": "#2ecc71",  # Green
    "apply_bc": "#f1c40f",         # Yellow
    "solve_system": "#e74c3c",     # Red
    "compute_derived": "#9b59b6",  # Purple
    "export_results": "#95a5a6",   # Gray
}

# Timeline event categories
CATEGORY_CUDA_KERNEL = "cuda_kernel"
CATEGORY_CUDA_MEMCPY_H2D = "cuda_memcpy_h2d"
CATEGORY_CUDA_MEMCPY_D2H = "cuda_memcpy_d2h"
CATEGORY_CUDA_MEMCPY_D2D = "cuda_memcpy_d2d"
CATEGORY_CUDA_SYNC = "cuda_sync"
CATEGORY_NVTX_RANGE = "nvtx_range"
CATEGORY_CPU_ACTIVITY = "cpu_activity"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimelineEvent:
    """Single event in the profiling timeline (Gantt-ready)."""
    id: str
    category: str
    name: str
    start_ns: int
    duration_ns: int
    end_ns: int
    stream: int = 0
    correlation_id: int = 0
    metadata: Optional[Dict[str, Any]] = None
    color: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.color is None:
            self.color = self._default_color()
    
    def _default_color(self) -> str:
        """Assign default color based on category."""
        colors = {
            CATEGORY_CUDA_KERNEL: "#e74c3c",
            CATEGORY_CUDA_MEMCPY_H2D: "#3498db",
            CATEGORY_CUDA_MEMCPY_D2H: "#2ecc71",
            CATEGORY_CUDA_MEMCPY_D2D: "#f39c12",
            CATEGORY_CUDA_SYNC: "#95a5a6",
            CATEGORY_NVTX_RANGE: "#9b59b6",
            CATEGORY_CPU_ACTIVITY: "#34495e",
        }
        return colors.get(self.category, "#7f8c8d")


@dataclass
class KernelMetrics:
    """Detailed metrics for a CUDA kernel from Nsight Compute."""
    kernel_name: str
    invocation: int = 1
    duration_ns: int = 0
    grid: Optional[List[int]] = None
    block: Optional[List[int]] = None
    registers_per_thread: int = 0
    shared_memory_static: int = 0
    shared_memory_dynamic: int = 0
    occupancy_achieved: float = 0.0
    occupancy_theoretical: float = 0.0
    sm_throughput_pct: float = 0.0
    dram_throughput_pct: float = 0.0
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    warp_execution_efficiency: float = 0.0
    
    def __post_init__(self):
        if self.grid is None:
            self.grid = [1, 1, 1]
        if self.block is None:
            self.block = [1, 1, 1]


@dataclass
class ProfileSession:
    """Profile session metadata."""
    id: str
    solver: str
    mesh: str
    mesh_file: str
    mode: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    nsys_file: Optional[str] = None
    ncu_file: Optional[str] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    linked_job_id: Optional[str] = None  # Link to original solve job
    timeline_summary: Optional[Dict[str, Any]] = None  # Cached summary stats
    mesh_nodes: Optional[int] = None
    mesh_elements: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Profiling Service
# =============================================================================

class ProfilingService:
    """Service for managing NVIDIA Nsight profiling sessions."""
    
    def __init__(self, profiles_dir: Path, benchmark_script: Optional[Path] = None):
        """
        Initialize profiling service.
        
        Args:
            profiles_dir: Directory for storing profile outputs
            benchmark_script: Path to benchmark runner script
        """
        self.profiles_dir = Path(profiles_dir)
        self.nsys_dir = self.profiles_dir / "nsys"
        self.ncu_dir = self.profiles_dir / "ncu"
        self.sessions_file = self.profiles_dir / "sessions.json"
        self.benchmark_script = benchmark_script
        
        # Ensure directories exist
        self.nsys_dir.mkdir(parents=True, exist_ok=True)
        self.ncu_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing sessions
        self.sessions: Dict[str, ProfileSession] = {}
        self._load_sessions()
        
        # Track running processes
        self._running: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        
        # Event callback for Socket.IO notifications
        self._event_callback: Optional[Callable[[str, str, dict], None]] = None
        
        # Check tool availability
        self.nsys_available = self._check_tool("nsys", "--version")
        self.ncu_available = self._check_tool("ncu", "--version")
        
        print(f"[Profiling] Initialized - nsys: {self.nsys_available}, ncu: {self.ncu_available}")
    
    def set_event_callback(self, callback: Optional[Callable[[str, str, dict], None]]):
        """
        Set callback for profiling events.
        
        Args:
            callback: Function(event_type, session_id, data) called on events.
                     event_type: 'started', 'progress', 'complete', 'error'
        """
        self._event_callback = callback
    
    def _emit_event(self, event_type: str, session_id: str, **data):
        """
        Emit a profiling event via callback.
        
        Args:
            event_type: Type of event ('started', 'progress', 'complete', 'error')
            session_id: Session identifier
            **data: Additional event data
        """
        if self._event_callback:
            try:
                self._event_callback(event_type, session_id, data)
            except Exception as e:
                print(f"[Profiling] Event callback error: {e}")
    
    def _check_tool(self, tool: str, arg: str) -> bool:
        """Check if a profiling tool is available."""
        try:
            result = subprocess.run(
                [tool, arg],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _load_sessions(self):
        """Load sessions from JSON file."""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    for sid, sdata in data.items():
                        self.sessions[sid] = ProfileSession(**sdata)
                print(f"[Profiling] Loaded {len(self.sessions)} sessions")
            except Exception as e:
                print(f"[Profiling] Error loading sessions: {e}")
    
    def _save_sessions(self):
        """Save sessions to JSON file."""
        try:
            data = {sid: s.to_dict() for sid, s in self.sessions.items()}
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Profiling] Error saving sessions: {e}")
    
    def get_available_modes(self) -> Dict[str, Any]:
        """Get available profiling modes based on installed tools."""
        return {
            "modes": [
                {
                    "id": ProfileMode.TIMELINE.value,
                    "name": "Timeline",
                    "description": "GPU activity timeline (fast)",
                    "available": self.nsys_available
                },
                {
                    "id": ProfileMode.KERNELS.value,
                    "name": "Kernel Metrics",
                    "description": "Detailed kernel analysis (slow)",
                    "available": self.ncu_available
                },
                {
                    "id": ProfileMode.FULL.value,
                    "name": "Full Analysis",
                    "description": "Timeline + kernel metrics",
                    "available": self.nsys_available and self.ncu_available
                }
            ],
            "nsys_available": self.nsys_available,
            "ncu_available": self.ncu_available
        }
    
    def start_profiled_run(
        self,
        solver: str,
        mesh_file: str,
        mode: str = ProfileMode.TIMELINE.value,
        benchmark_args: Optional[Dict[str, Any]] = None,
        mesh_nodes: Optional[int] = None,
        mesh_elements: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start a profiled benchmark run.
        
        Args:
            solver: Solver type (gpu, numba_cuda, etc.)
            mesh_file: Path to mesh file
            mode: Profile mode (timeline, kernels, full)
            benchmark_args: Additional arguments for benchmark script
            
        Returns:
            Dict with session_id and status
        """
        # Validate mode
        if mode == ProfileMode.TIMELINE.value and not self.nsys_available:
            return {"error": "Nsight Systems (nsys) not available"}
        if mode == ProfileMode.KERNELS.value and not self.ncu_available:
            return {"error": "Nsight Compute (ncu) not available"}
        if mode == ProfileMode.FULL.value and not (self.nsys_available and self.ncu_available):
            return {"error": "Both nsys and ncu required for full mode"}
        
        # Create session
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        mesh_name = Path(mesh_file).stem
        
        # Extract linked_job_id if provided
        linked_job_id = None
        if benchmark_args:
            linked_job_id = benchmark_args.pop('linked_job_id', None)
        
        session = ProfileSession(
            id=session_id,
            solver=solver,
            mesh=mesh_name,
            mesh_file=mesh_file,
            mode=mode,
            status=SessionStatus.PENDING.value,
            created_at=timestamp,
            linked_job_id=linked_job_id,
            mesh_nodes=mesh_nodes,
            mesh_elements=mesh_elements
        )
        
        with self._lock:
            self.sessions[session_id] = session
            self._save_sessions()
        
        # Emit started event
        self._emit_event('started', session_id,
            status=SessionStatus.PENDING.value,
            solver=solver,
            mesh=mesh_name,
            mode=mode
        )
        
        # Start profiling in background thread
        thread = threading.Thread(
            target=self._run_profiling,
            args=(session_id, solver, mesh_file, mode, benchmark_args),
            daemon=True
        )
        thread.start()
        
        return {
            "session_id": session_id,
            "status": SessionStatus.PENDING.value,
            "mode": mode
        }
    
    def _run_profiling(
        self,
        session_id: str,
        solver: str,
        mesh_file: str,
        mode: str,
        benchmark_args: Dict[str, Any]
    ):
        """Execute profiling (runs in background thread)."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            # Update status to running
            session.status = SessionStatus.RUNNING.value
            self._save_sessions()
            
            # Emit progress event
            self._emit_event('progress', session_id,
                status=SessionStatus.RUNNING.value,
                stage='starting',
                message='Starting profiler...'
            )
            
            # Build base benchmark command
            if self.benchmark_script and self.benchmark_script.exists():
                base_cmd = ["python", str(self.benchmark_script)]
            else:
                # Fallback: assume run_single_benchmark.py in current dir
                base_cmd = ["python", "run_single_benchmark.py"]
            
            base_cmd.extend(["--solver", solver, "--mesh", mesh_file, "--quiet"])
            
            if benchmark_args:
                for key, value in benchmark_args.items():
                    base_cmd.extend([f"--{key}", str(value)])
            
            # Run nsys if needed
            if mode in (ProfileMode.TIMELINE.value, ProfileMode.FULL.value):
                self._emit_event('progress', session_id,
                    status=SessionStatus.RUNNING.value,
                    stage='nsys',
                    message='Running Nsight Systems profiler...'
                )
                
                nsys_output = self.nsys_dir / f"{session_id}"
                nsys_cmd = [
                    "nsys", "profile",
                    "-o", str(nsys_output),
                    "--trace", "cuda,nvtx",
                    "--force-overwrite", "true",
                    "--export", "sqlite"
                ] + base_cmd
                
                print(f"[Profiling] Running nsys: {' '.join(nsys_cmd)}")
                result = subprocess.run(nsys_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"nsys failed: {result.stderr}")
                
                session.nsys_file = f"{session_id}.nsys-rep"
            
            # Run ncu if needed
            if mode in (ProfileMode.KERNELS.value, ProfileMode.FULL.value):
                self._emit_event('progress', session_id,
                    status=SessionStatus.RUNNING.value,
                    stage='ncu',
                    message='Running Nsight Compute profiler...'
                )
                
                ncu_output = self.ncu_dir / f"{session_id}.ncu-rep"
                ncu_csv = self.ncu_dir / f"{session_id}.csv"
                
                ncu_cmd = [
                    "ncu",
                    "--set", "full",
                    "-o", str(ncu_output),
                    "--csv"
                ] + base_cmd
                
                print(f"[Profiling] Running ncu: {' '.join(ncu_cmd)}")
                result = subprocess.run(ncu_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"ncu failed: {result.stderr}")
                
                # Save CSV output
                if result.stdout:
                    with open(ncu_csv, 'w') as f:
                        f.write(result.stdout)
                
                session.ncu_file = f"{session_id}.ncu-rep"
            
            # Extract metrics
            session.status = SessionStatus.EXTRACTING.value
            self._save_sessions()
            
            self._emit_event('progress', session_id,
                status=SessionStatus.EXTRACTING.value,
                stage='extracting',
                message='Extracting profiling data...'
            )
            
            # Extraction happens on-demand via get_timeline/get_kernels
            
            session.status = SessionStatus.COMPLETED.value
            session.completed_at = datetime.now().isoformat()
            self._save_sessions()
            
            # Emit complete event
            self._emit_event('complete', session_id,
                status=SessionStatus.COMPLETED.value,
                message='Profiling complete'
            )
            
        except Exception as e:
            print(f"[Profiling] Error in session {session_id}: {e}")
            session.status = SessionStatus.FAILED.value
            session.error = str(e)
            self._save_sessions()
            
            # Emit error event
            self._emit_event('error', session_id,
                status=SessionStatus.FAILED.value,
                error=str(e),
                message=f'Profiling failed: {e}'
            )
    
    def get_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of profile sessions (most recent first)."""
        sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.created_at,
            reverse=True
        )[:limit]
        return [s.to_dict() for s in sessions]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata and summary."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        result = session.to_dict()
        
        # Add file availability flags
        if session.nsys_file:
            result["timeline_available"] = (self.nsys_dir / f"{session_id}.sqlite").exists()
        if session.ncu_file:
            result["kernels_available"] = (self.ncu_dir / f"{session_id}.csv").exists()
        
        return result
    
    def generate_timeline_hdf5(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate client-optimized HDF5 timeline from nsys export.
        
        Uses GPU-accelerated transformer for fast processing.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with generation statistics, or None if session not found
        """
        session = self.sessions.get(session_id)
        if not session or not session.nsys_file:
            return None
        
        nsys_rep = self.nsys_dir / session.nsys_file
        if not nsys_rep.exists():
            return {"error": f"Nsight report not found: {session.nsys_file}"}
        
        h5_path = self.nsys_dir / f"{session_id}.h5"
        nsys_h5_path = self.nsys_dir / f"{session_id}.nsys.h5"
        
        # Check if already generated
        if h5_path.exists():
            # If summary not cached yet, we still return cached status
            # Summary will be populated on next generation or can be extracted from HDF5
            return {
                "session_id": session_id,
                "status": "cached",
                "h5_path": str(h5_path),
                "file_size_bytes": h5_path.stat().st_size,
                "timeline_summary": session.timeline_summary
            }
        
        if ProfilingHDF5Transformer is None:
            return {"error": "HDF5 transformer not available"}
        
        total_start = time.perf_counter()
        
        # Step 1: Export nsys-rep to native HDF5
        if not nsys_h5_path.exists():
            export_start = time.perf_counter()
            self._export_nsys_hdf5(nsys_rep, nsys_h5_path)
            export_time = time.perf_counter() - export_start
            print(f"[Profiling] Export to nsys HDF5: {export_time:.3f}s")
        else:
            print(f"[Profiling] Using cached nsys HDF5 export")
        
        if not nsys_h5_path.exists():
            return {"error": "Failed to export nsys HDF5"}
        
        # Step 2: Transform to client-optimized HDF5 using GPU
        try:
            transformer = ProfilingHDF5Transformer(use_gpu=True)
            stats = transformer.transform(nsys_h5_path, h5_path, session_id)
        except Exception as e:
            print(f"[Profiling] HDF5 transform failed: {e}")
            return {"error": f"HDF5 transform failed: {e}"}
        
        total_time = time.perf_counter() - total_start
        
        print(f"[Profiling] Timeline HDF5 generated: {stats['total_events']} events in {total_time:.3f}s")
        print(f"[Profiling]   File size: {stats['file_size_bytes'] / 1024:.1f} KB")
        print(f"[Profiling]   GPU used: {stats['used_gpu']}")
        for cat, cat_stats in stats['categories'].items():
            print(f"[Profiling]   {cat}: {cat_stats['count']} events")
        
        # Store summary in session for quick access in session list
        session.timeline_summary = {
            'total_duration_ms': stats['total_duration_ns'] / 1e6,
            'total_events': stats['total_events'],
            'categories': stats['categories']
        }
        with self._lock:
            self._save_sessions()
        
        return {
            "session_id": session_id,
            "status": "generated",
            "h5_path": str(h5_path),
            "total_events": stats['total_events'],
            "total_duration_ns": stats['total_duration_ns'],
            "file_size_bytes": stats['file_size_bytes'],
            "generation_time_s": total_time,
            "used_gpu": stats['used_gpu'],
            "categories": stats['categories'],
            "timings": stats['timings']
        }
    
    def _export_nsys_hdf5(self, nsys_rep: Path, hdf5_file: Path):
        """Export .nsys-rep to native HDF5 format."""
        try:
            cmd = [
                "nsys", "export",
                "--type", "hdf",
                "--output", str(hdf5_file),
                "--force-overwrite", "true",
                str(nsys_rep)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[Profiling] HDF5 export error: {result.stderr}")
            else:
                print(f"[Profiling] Exported HDF5: {hdf5_file}")
        except Exception as e:
            print(f"[Profiling] HDF5 export failed: {e}")
    
    def get_kernels(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract kernel metrics from Nsight Compute CSV output.
        """
        session = self.sessions.get(session_id)
        if not session or not session.ncu_file:
            return None
        
        csv_file = self.ncu_dir / f"{session_id}.csv"
        if not csv_file.exists():
            return {"error": "Kernel CSV not available"}
        
        kernels = []
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    kernel = KernelMetrics(
                        kernel_name=row.get('Kernel Name', 'unknown'),
                        invocation=int(row.get('Invocations', 1)),
                        duration_ns=int(float(row.get('Duration', 0)) * 1e9),
                        grid=[
                            int(row.get('Grid Size X', 1)),
                            int(row.get('Grid Size Y', 1)),
                            int(row.get('Grid Size Z', 1))
                        ],
                        block=[
                            int(row.get('Block Size X', 1)),
                            int(row.get('Block Size Y', 1)),
                            int(row.get('Block Size Z', 1))
                        ],
                        registers_per_thread=int(row.get('Registers Per Thread', 0)),
                        shared_memory_static=int(row.get('Static SMem Per Block', 0)),
                        shared_memory_dynamic=int(row.get('Dynamic SMem Per Block', 0)),
                        occupancy_achieved=float(row.get('Achieved Occupancy', 0)),
                        occupancy_theoretical=float(row.get('Theoretical Occupancy', 0)),
                        sm_throughput_pct=float(row.get('SM [%]', 0)),
                        dram_throughput_pct=float(row.get('DRAM [%]', 0))
                    )
                    kernels.append(asdict(kernel))
        
        except Exception as e:
            print(f"[Profiling] Error parsing kernel CSV: {e}")
            return {"error": str(e)}
        
        return {
            "session_id": session_id,
            "kernels": kernels,
            "kernel_count": len(kernels)
        }
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a profile session and its files."""
        session = self.sessions.get(session_id)
        if not session:
            return {"deleted": False, "error": "Session not found"}
        
        # Delete files
        files_deleted = []
        
        if session.nsys_file:
            # Include all generated files in cleanup
            for ext in ['.nsys-rep', '.sqlite', '.h5', '.nsys.h5']:
                f = self.nsys_dir / f"{session_id}{ext}"
                if f.exists():
                    f.unlink()
                    files_deleted.append(str(f))
        
        if session.ncu_file:
            for ext in ['.ncu-rep', '.csv']:
                f = self.ncu_dir / f"{session_id}{ext}"
                if f.exists():
                    f.unlink()
                    files_deleted.append(str(f))
        
        # Remove from sessions
        with self._lock:
            del self.sessions[session_id]
            self._save_sessions()
        
        return {
            "deleted": True,
            "session_id": session_id,
            "files_deleted": files_deleted
        }
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> Dict[str, Any]:
        """Remove sessions older than max_age_days."""
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        deleted = []
        
        for session_id, session in list(self.sessions.items()):
            try:
                created = datetime.fromisoformat(session.created_at).timestamp()
                if created < cutoff:
                    self.delete_session(session_id)
                    deleted.append(session_id)
            except:
                pass
        
        return {
            "deleted_count": len(deleted),
            "deleted_sessions": deleted
        }


# =============================================================================
# FastAPI Router
# =============================================================================

def create_profiling_router(service: ProfilingService):
    """Create FastAPI router for profiling API endpoints."""
    from fastapi import APIRouter, HTTPException, Response, Request
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    from typing import Optional
    
    router = APIRouter(prefix="/api/profiling", tags=["profiling"])
    
    class RunProfileRequest(BaseModel):
        solver: str
        mesh_file: str
        mode: str = ProfileMode.TIMELINE.value
        benchmark_args: Optional[Dict[str, Any]] = None
        mesh_nodes: Optional[int] = None
        mesh_elements: Optional[int] = None
    
    @router.get("/modes")
    async def get_modes(response: Response):
        """Get available profiling modes."""
        response.headers["Cache-Control"] = "no-store"
        return service.get_available_modes()
    
    @router.post("/run")
    async def run_profile(request: RunProfileRequest):
        """Start a profiled benchmark run."""
        result = service.start_profiled_run(
            solver=request.solver,
            mesh_file=request.mesh_file,
            mode=request.mode,
            benchmark_args=request.benchmark_args or {},
            mesh_nodes=request.mesh_nodes,
            mesh_elements=request.mesh_elements
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    @router.get("/sessions")
    async def get_sessions(response: Response, limit: int = 50):
        """Get list of profile sessions."""
        response.headers["Cache-Control"] = "no-store"
        return {"sessions": service.get_sessions(limit)}
    
    @router.get("/session/{session_id}")
    async def get_session(session_id: str, response: Response):
        """Get session metadata and summary."""
        response.headers["Cache-Control"] = "no-store"
        result = service.get_session(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
    
    @router.api_route("/timeline/{session_id}.h5", methods=["GET", "HEAD"])
    async def get_timeline_hdf5(session_id: str, request: Request):
        """
        Get timeline data in HDF5 format (optimized binary).
        
        Returns the pre-generated HDF5 file for fast client loading.
        Generates on-demand using GPU-accelerated transformer if not present.
        """
        h5_path = service.nsys_dir / f"{session_id}.h5"
        
        # Generate if not exists
        if not h5_path.exists():
            result = service.generate_timeline_hdf5(session_id)
            if not result:
                raise HTTPException(status_code=404, detail="Session not found")
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
        
        if not h5_path.exists():
            raise HTTPException(status_code=404, detail="HDF5 file not available")
        
        return FileResponse(
            path=h5_path,
            media_type="application/x-hdf5",
            filename=f"{session_id}.h5",
            headers={
                "Cache-Control": "public, max-age=31536000",
            }
        )
    
    @router.get("/timeline/{session_id}/status")
    async def get_timeline_status(session_id: str, response: Response):
        """
        Get timeline generation status and metadata.
        
        Returns info about the HDF5 file without downloading it.
        Triggers generation if not already present.
        """
        response.headers["Cache-Control"] = "no-store"
        result = service.generate_timeline_hdf5(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    
    @router.get("/kernels/{session_id}")
    async def get_kernels(session_id: str, response: Response):
        """Get kernel metrics from Nsight Compute."""
        response.headers["Cache-Control"] = "no-store"
        result = service.get_kernels(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found or no kernel data")
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    
    @router.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        """Delete a profile session."""
        result = service.delete_session(session_id)
        if not result.get("deleted"):
            raise HTTPException(status_code=404, detail=result.get("error", "Delete failed"))
        return result
    
    @router.post("/cleanup")
    async def cleanup(max_age_days: int = 30):
        """Remove old profile sessions."""
        return service.cleanup_old_sessions(max_age_days)

    @router.get("/report/{session_id}")
    async def download_report(session_id: str):
        """
        Download the original Nsight Systems report file (.nsys-rep).
        
        This file can be opened in NVIDIA Nsight Systems desktop application.
        """
        session = service.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not session.nsys_file:
            raise HTTPException(status_code=404, detail="No report file for this session")
        
        report_path = service.nsys_dir / session.nsys_file
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found on disk")
        
        return FileResponse(
            path=report_path,
            media_type="application/octet-stream",
            filename=session.nsys_file,
            headers={
                "Content-Disposition": f'attachment; filename="{session.nsys_file}"'
            }
        )

    return router


# =============================================================================
# Module Initialization
# =============================================================================

def init_profiling_service(
    profiles_dir: Path | str,
    benchmark_script: Optional[Path | str] = None
) -> ProfilingService:
    """
    Initialize profiling service.
    
    Args:
        profiles_dir: Directory for profile outputs
        benchmark_script: Path to benchmark runner script
        
    Returns:
        ProfilingService instance
    """
    return ProfilingService(
        Path(profiles_dir),
        Path(benchmark_script) if benchmark_script else None
    )


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=== Profiling Service Test ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ProfilingService(Path(tmpdir))
        
        # Check available modes
        modes = service.get_available_modes()
        print(f"Available modes: {json.dumps(modes, indent=2)}")
        
        # List sessions (should be empty)
        sessions = service.get_sessions()
        print(f"\nSessions: {sessions}")
        
        print("\n[Note] Full test requires nsys/ncu and a benchmark script")
