"""
Profiling Service - NVIDIA Nsight Systems and Compute integration.

This service:
- Executes benchmark runs wrapped with nsys/ncu profiling
- Extracts metrics from Nsight SQLite exports and CSV outputs
- Provides timeline data suitable for Gantt visualization
- Manages profile session storage and cleanup

Location: /src/app/server/profiling_service.py
Data directory: /data/profiles/

Dependencies:
- NVIDIA Nsight Systems (nsys) in PATH
- NVIDIA Nsight Compute (ncu) in PATH
- sqlite3 (standard library)
"""

import json
import sqlite3
import subprocess
import csv
import shutil
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


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
        
        # Check tool availability
        self.nsys_available = self._check_tool("nsys", "--version")
        self.ncu_available = self._check_tool("ncu", "--version")
        
        print(f"[Profiling] Initialized - nsys: {self.nsys_available}, ncu: {self.ncu_available}")
    
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
                    "name": "Timeline (nsys)",
                    "description": "System-wide timeline - low overhead, shows kernel launches, memory transfers, NVTX ranges",
                    "available": self.nsys_available,
                    "overhead": "low"
                },
                {
                    "id": ProfileMode.KERNELS.value,
                    "name": "Kernel Analysis (ncu)",
                    "description": "Deep kernel metrics - high overhead, shows occupancy, throughput, stalls",
                    "available": self.ncu_available,
                    "overhead": "high"
                },
                {
                    "id": ProfileMode.FULL.value,
                    "name": "Full Analysis",
                    "description": "Both timeline and kernel analysis sequentially",
                    "available": self.nsys_available and self.ncu_available,
                    "overhead": "high"
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
        benchmark_args: Optional[Dict[str, Any]] = None
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
        
        session = ProfileSession(
            id=session_id,
            solver=solver,
            mesh=mesh_name,
            mesh_file=mesh_file,
            mode=mode,
            status=SessionStatus.PENDING.value,
            created_at=timestamp
        )
        
        with self._lock:
            self.sessions[session_id] = session
            self._save_sessions()
        
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
            session.status = SessionStatus.RUNNING.value
            self._save_sessions()
            
            # Build base benchmark command
            if self.benchmark_script and self.benchmark_script.exists():
                base_cmd = ["python", str(self.benchmark_script)]
            else:
                # Fallback: assume run_benchmark.py in current dir
                base_cmd = ["python", "run_benchmark.py"]
            
            base_cmd.extend(["--solver", solver, "--mesh", mesh_file])
            
            if benchmark_args:
                for key, value in benchmark_args.items():
                    base_cmd.extend([f"--{key}", str(value)])
            
            # Run nsys if needed
            if mode in (ProfileMode.TIMELINE.value, ProfileMode.FULL.value):
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
            
            # Extraction happens on-demand via get_timeline/get_kernels
            
            session.status = SessionStatus.COMPLETED.value
            session.completed_at = datetime.now().isoformat()
            
        except Exception as e:
            print(f"[Profiling] Error in session {session_id}: {e}")
            session.status = SessionStatus.FAILED.value
            session.error = str(e)
        
        finally:
            self._save_sessions()
    
    def get_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of profile sessions."""
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
        
        # Add summary stats if completed
        if session.status == SessionStatus.COMPLETED.value:
            if session.nsys_file:
                timeline = self.get_timeline(session_id)
                if timeline:
                    result["timeline_summary"] = {
                        "total_events": len(timeline.get("events", [])),
                        "categories": self._count_categories(timeline.get("events", [])),
                        "total_duration_ms": timeline.get("total_duration_ns", 0) / 1e6
                    }
            
            if session.ncu_file:
                kernels = self.get_kernels(session_id)
                if kernels:
                    result["kernel_summary"] = {
                        "total_kernels": len(kernels.get("kernels", [])),
                        "kernel_names": [k["kernel_name"] for k in kernels.get("kernels", [])]
                    }
        
        return result
    
    def _count_categories(self, events: List[Dict]) -> Dict[str, int]:
        """Count events by category."""
        counts = {}
        for event in events:
            cat = event.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    def get_timeline(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract timeline data from Nsight Systems SQLite export.
        
        Returns Gantt-ready timeline events.
        """
        session = self.sessions.get(session_id)
        if not session or not session.nsys_file:
            return None
        
        sqlite_file = self.nsys_dir / f"{session_id}.sqlite"
        if not sqlite_file.exists():
            # Try to export if .nsys-rep exists
            nsys_rep = self.nsys_dir / session.nsys_file
            if nsys_rep.exists():
                self._export_nsys_sqlite(nsys_rep, sqlite_file)
        
        if not sqlite_file.exists():
            return {"error": "SQLite export not available"}
        
        events = []
        min_start = float('inf')
        max_end = 0
        
        try:
            conn = sqlite3.connect(str(sqlite_file))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Extract CUDA kernels
            try:
                cursor.execute("""
                    SELECT 
                        k.correlationId,
                        k.start,
                        k.end,
                        k.deviceId,
                        k.streamId,
                        k.gridX, k.gridY, k.gridZ,
                        k.blockX, k.blockY, k.blockZ,
                        k.registersPerThread,
                        k.staticSharedMemory,
                        k.dynamicSharedMemory,
                        s.value as name
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    LEFT JOIN StringIds s ON k.demangledName = s.id
                    ORDER BY k.start
                """)
                
                for row in cursor.fetchall():
                    start = row['start']
                    end = row['end']
                    min_start = min(min_start, start)
                    max_end = max(max_end, end)
                    
                    event = TimelineEvent(
                        id=f"kernel_{row['correlationId']}",
                        category=CATEGORY_CUDA_KERNEL,
                        name=row['name'] or "unknown_kernel",
                        start_ns=start,
                        duration_ns=end - start,
                        end_ns=end,
                        stream=row['streamId'] or 0,
                        correlation_id=row['correlationId'],
                        metadata={
                            "grid": [row['gridX'], row['gridY'], row['gridZ']],
                            "block": [row['blockX'], row['blockY'], row['blockZ']],
                            "registers_per_thread": row['registersPerThread'],
                            "shared_memory_static": row['staticSharedMemory'],
                            "shared_memory_dynamic": row['dynamicSharedMemory'],
                            "device_id": row['deviceId']
                        }
                    )
                    events.append(asdict(event))
            except sqlite3.OperationalError as e:
                print(f"[Profiling] Kernel query failed: {e}")
            
            # Extract memory transfers
            try:
                cursor.execute("""
                    SELECT 
                        correlationId,
                        start,
                        end,
                        bytes,
                        copyKind,
                        streamId
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    ORDER BY start
                """)
                
                for row in cursor.fetchall():
                    start = row['start']
                    end = row['end']
                    min_start = min(min_start, start)
                    max_end = max(max_end, end)
                    
                    # Determine category based on copyKind
                    copy_kind = row['copyKind']
                    if copy_kind == 1:  # HtoD
                        category = CATEGORY_CUDA_MEMCPY_H2D
                        name = "cudaMemcpy HtoD"
                    elif copy_kind == 2:  # DtoH
                        category = CATEGORY_CUDA_MEMCPY_D2H
                        name = "cudaMemcpy DtoH"
                    else:
                        category = CATEGORY_CUDA_MEMCPY_D2D
                        name = f"cudaMemcpy (kind={copy_kind})"
                    
                    event = TimelineEvent(
                        id=f"memcpy_{row['correlationId']}",
                        category=category,
                        name=name,
                        start_ns=start,
                        duration_ns=end - start,
                        end_ns=end,
                        stream=row['streamId'] or 0,
                        correlation_id=row['correlationId'],
                        metadata={
                            "bytes": row['bytes'],
                            "throughput_gbps": row['bytes'] / (end - start) if end > start else 0
                        }
                    )
                    events.append(asdict(event))
            except sqlite3.OperationalError as e:
                print(f"[Profiling] Memcpy query failed: {e}")
            
            # Extract NVTX ranges (pipeline stages)
            try:
                cursor.execute("""
                    SELECT 
                        r.start,
                        r.end,
                        s.value as text,
                        r.category
                    FROM NVTX_EVENTS r
                    LEFT JOIN StringIds s ON r.text = s.id
                    WHERE r.start IS NOT NULL AND r.end IS NOT NULL
                    ORDER BY r.start
                """)
                
                nvtx_id = 0
                for row in cursor.fetchall():
                    start = row['start']
                    end = row['end']
                    if start and end:
                        min_start = min(min_start, start)
                        max_end = max(max_end, end)
                        
                        name = row['text'] or "nvtx_range"
                        color = NVTX_COLORS.get(name, "#9b59b6")
                        
                        event = TimelineEvent(
                            id=f"nvtx_{nvtx_id}",
                            category=CATEGORY_NVTX_RANGE,
                            name=name,
                            start_ns=start,
                            duration_ns=end - start,
                            end_ns=end,
                            color=color,
                            metadata={
                                "category_id": row['category']
                            }
                        )
                        events.append(asdict(event))
                        nvtx_id += 1
            except sqlite3.OperationalError as e:
                print(f"[Profiling] NVTX query failed: {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"[Profiling] Error extracting timeline: {e}")
            return {"error": str(e)}
        
        # Sort by start time
        events.sort(key=lambda e: e['start_ns'])
        
        # Normalize timestamps to start from 0
        if events and min_start != float('inf'):
            for event in events:
                event['start_ns'] -= min_start
                event['end_ns'] -= min_start
        
        return {
            "session_id": session_id,
            "events": events,
            "total_duration_ns": max_end - min_start if max_end > min_start else 0,
            "event_count": len(events),
            "categories": self._count_categories(events)
        }
    
    def _export_nsys_sqlite(self, nsys_rep: Path, sqlite_file: Path):
        """Export .nsys-rep to SQLite format."""
        try:
            cmd = [
                "nsys", "export",
                "--type", "sqlite",
                "--output", str(sqlite_file),
                str(nsys_rep)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"[Profiling] Exported SQLite: {sqlite_file}")
        except Exception as e:
            print(f"[Profiling] SQLite export failed: {e}")
    
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
                # Skip header lines that start with ==
                lines = [l for l in f.readlines() if not l.startswith('==')]
                
            if not lines:
                return {"kernels": []}
            
            reader = csv.DictReader(lines)
            
            for row in reader:
                kernel = KernelMetrics(
                    kernel_name=row.get('Kernel Name', 'unknown'),
                    invocation=int(row.get('Invocation', 1) or 1),
                    duration_ns=int(float(row.get('Duration', 0) or 0) * 1e9),
                    registers_per_thread=int(row.get('Registers Per Thread', 0) or 0),
                    occupancy_achieved=float(row.get('Achieved Occupancy', 0) or 0),
                    occupancy_theoretical=float(row.get('Theoretical Occupancy', 0) or 0),
                    sm_throughput_pct=float(row.get('SM [%]', 0) or 0),
                    dram_throughput_pct=float(row.get('DRAM [%]', 0) or 0),
                )
                
                # Parse grid/block if available
                if 'Grid Size' in row:
                    try:
                        grid_str = row['Grid Size'].strip('()').split(',')
                        kernel.grid = [int(x.strip()) for x in grid_str]
                    except:
                        pass
                
                if 'Block Size' in row:
                    try:
                        block_str = row['Block Size'].strip('()').split(',')
                        kernel.block = [int(x.strip()) for x in block_str]
                    except:
                        pass
                
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
            for ext in ['.nsys-rep', '.sqlite']:
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
    from fastapi import APIRouter, HTTPException, Response
    from pydantic import BaseModel
    from typing import Optional
    
    router = APIRouter(prefix="/api/profiling", tags=["profiling"])
    
    class RunProfileRequest(BaseModel):
        solver: str
        mesh_file: str
        mode: str = ProfileMode.TIMELINE.value
        benchmark_args: Optional[Dict[str, Any]] = None
    
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
            benchmark_args=request.benchmark_args or {}
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
    
    @router.get("/timeline/{session_id}")
    async def get_timeline(session_id: str, response: Response):
        """Get timeline data for Gantt visualization."""
        response.headers["Cache-Control"] = "no-store"
        result = service.get_timeline(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found or no timeline data")
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
