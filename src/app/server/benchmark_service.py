"""
Benchmark Service - Records and serves FEM solver benchmark results.

This service:
- Subscribes to Socket.IO solver events (solve_complete)
- Auto-detects server hardware configuration
- Records benchmark results with timestamps
- Persists data to JSON file
- Provides REST API for benchmark data access

Location: /src/app/server/benchmark_service.py
Data file: /src/app/server/benchmark/benchmark_results.json
"""

import os
import json
import uuid
import hashlib
import platform
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import asyncio


# =============================================================================
# Server Hardware Detection
# =============================================================================

def detect_server_hardware() -> Dict[str, Any]:
    """Detect server hardware configuration."""
    config = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": "Unknown",
        "cpu_cores": os.cpu_count() or 0,
        "ram_gb": 0,
        "gpu_model": None,
        "gpu_memory_gb": None,
        "cuda_version": None,
    }
    
    # CPU model detection (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    config["cpu_model"] = line.split(':')[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass
    
    # RAM detection (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    config["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
                    break
    except (FileNotFoundError, PermissionError):
        pass
    
    # GPU detection via NVIDIA tools
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 2:
                config["gpu_model"] = parts[0]
                config["gpu_memory_gb"] = round(int(parts[1]) / 1024, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # CUDA version detection
    try:
        import subprocess
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Extract version like "release 12.1"
                    parts = line.split('release')
                    if len(parts) > 1:
                        config["cuda_version"] = parts[1].strip().split(',')[0].strip()
                        break
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # Alternative: CuPy CUDA detection
    if not config["cuda_version"]:
        try:
            import cupy as cp
            config["cuda_version"] = f"{cp.cuda.runtime.runtimeGetVersion() // 1000}.{(cp.cuda.runtime.runtimeGetVersion() % 1000) // 10}"
        except Exception:
            pass
    
    return config


def generate_config_hash(config: Dict[str, Any], keys: List[str]) -> str:
    """Generate a hash from specific config keys for grouping."""
    hash_input = "|".join(str(config.get(k, "")) for k in sorted(keys))
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


# =============================================================================
# Benchmark Record Structure
# =============================================================================

@dataclass
class BenchmarkRecord:
    """Single benchmark result record."""
    id: str
    timestamp: str
    
    # Model info
    model_file: str
    model_name: str
    model_nodes: int
    model_elements: int
    
    # Solver info
    solver_type: str
    
    # Results
    converged: bool
    iterations: int
    timings: Dict[str, float]
    solution_stats: Dict[str, Any]
    
    # Server configuration
    server_config: Dict[str, Any]
    server_hash: str
    
    # Client configuration
    client_config: Dict[str, Any]
    client_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkRecord':
        return cls(**data)


# =============================================================================
# Benchmark Service
# =============================================================================

class BenchmarkService:
    """
    Service for recording and serving benchmark results.
    
    Features:
    - In-memory cache for fast access
    - JSON file persistence
    - Auto-reload on file changes
    - Thread-safe operations
    """
    
    # Keys used for server config hashing
    SERVER_HASH_KEYS = ["hostname", "cpu_model", "cpu_cores", "ram_gb", "gpu_model", "gpu_memory_gb"]
    
    # Keys used for client config hashing
    CLIENT_HASH_KEYS = ["browser", "os", "gpu_vendor", "gpu_renderer"]
    
    def __init__(self, data_file: Path | str):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._records: Dict[str, BenchmarkRecord] = {}
        self._file_mtime: float = 0
        
        # Detect server hardware once at startup
        self.server_config = detect_server_hardware()
        self.server_hash = generate_config_hash(self.server_config, self.SERVER_HASH_KEYS)
        
        print(f"[Benchmark] Server config hash: {self.server_hash}")
        print(f"[Benchmark] CPU: {self.server_config['cpu_model']} ({self.server_config['cpu_cores']} cores)")
        print(f"[Benchmark] RAM: {self.server_config['ram_gb']} GB")
        if self.server_config['gpu_model']:
            print(f"[Benchmark] GPU: {self.server_config['gpu_model']} ({self.server_config['gpu_memory_gb']} GB)")
        
        # Load existing data
        self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load benchmark data from JSON file."""
        with self._lock:
            if not self.data_file.exists():
                self._records = {}
                self._save_to_file()
                return
            
            try:
                mtime = self.data_file.stat().st_mtime
                if mtime == self._file_mtime:
                    return  # No changes
                
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                self._records = {
                    r['id']: BenchmarkRecord.from_dict(r)
                    for r in data.get('records', [])
                }
                self._file_mtime = mtime
                
                print(f"[Benchmark] Loaded {len(self._records)} records from {self.data_file}")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Benchmark] Error loading data file: {e}")
                self._records = {}
    
    def _save_to_file(self) -> None:
        """Save benchmark data to JSON file."""
        with self._lock:
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "server_config": self.server_config,
                "records": [r.to_dict() for r in self._records.values()]
            }
            
            # Write atomically
            temp_file = self.data_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.data_file)
            
            self._file_mtime = self.data_file.stat().st_mtime
    
    def refresh(self) -> None:
        """Refresh data from file if changed externally."""
        self._load_from_file()
    
    def add_record(
        self,
        model_file: str,
        model_name: str,
        model_nodes: int,
        model_elements: int,
        solver_type: str,
        converged: bool,
        iterations: int,
        timings: Dict[str, float],
        solution_stats: Dict[str, Any],
        client_config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkRecord:
        """
        Add a new benchmark record.
        
        Args:
            model_file: Mesh filename
            model_name: Display name for model
            model_nodes: Number of nodes
            model_elements: Number of elements
            solver_type: Solver implementation used
            converged: Whether solver converged
            iterations: Number of iterations
            timings: Timing metrics dict
            solution_stats: Solution statistics
            client_config: Client hardware config (from browser)
        
        Returns:
            Created BenchmarkRecord
        """
        client_config = client_config or {}
        client_hash = generate_config_hash(client_config, self.CLIENT_HASH_KEYS) if client_config else "unknown"
        
        record = BenchmarkRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_file=model_file,
            model_name=model_name,
            model_nodes=model_nodes,
            model_elements=model_elements,
            solver_type=solver_type,
            converged=converged,
            iterations=iterations,
            timings=timings,
            solution_stats=solution_stats,
            server_config=self.server_config,
            server_hash=self.server_hash,
            client_config=client_config,
            client_hash=client_hash
        )
        
        with self._lock:
            self._records[record.id] = record
            self._save_to_file()
        
        print(f"[Benchmark] Added record {record.id[:8]}... "
              f"({model_name}, {solver_type}, {timings.get('total_program_time', 0):.2f}s)")
        
        return record
    
    def get_record(self, record_id: str) -> Optional[BenchmarkRecord]:
        """Get a single record by ID."""
        with self._lock:
            return self._records.get(record_id)
    
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        with self._lock:
            if record_id in self._records:
                del self._records[record_id]
                self._save_to_file()
                print(f"[Benchmark] Deleted record {record_id[:8]}...")
                return True
            return False
    
    def get_all_records(
        self,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all records with optional sorting and filtering.
        
        Args:
            sort_by: Field to sort by
            sort_order: 'asc' or 'desc'
            filters: Optional filters (e.g., {"solver_type": "gpu"})
        
        Returns:
            List of record dicts
        """
        with self._lock:
            records = [r.to_dict() for r in self._records.values()]
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                records = [r for r in records if r.get(key) == value]
        
        # Sort
        reverse = sort_order.lower() == "desc"
        
        def get_sort_key(record):
            if sort_by == "timestamp":
                return record.get("timestamp", "")
            elif sort_by == "total_time":
                return record.get("timings", {}).get("total_program_time", float('inf'))
            elif sort_by == "solver_type":
                return record.get("solver_type", "")
            elif sort_by == "model_name":
                return record.get("model_name", "")
            elif sort_by == "model_elements":
                return record.get("model_elements", 0)
            elif sort_by == "iterations":
                return record.get("iterations", 0)
            elif sort_by.startswith("timings."):
                timing_key = sort_by.split(".", 1)[1]
                return record.get("timings", {}).get(timing_key, float('inf'))
            else:
                return record.get(sort_by, "")
        
        records.sort(key=get_sort_key, reverse=reverse)
        
        return records
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of benchmark data."""
        with self._lock:
            records = list(self._records.values())
        
        if not records:
            return {
                "total_records": 0,
                "solver_types": [],
                "models": [],
                "server_hash": self.server_hash
            }
        
        solver_types = list(set(r.solver_type for r in records))
        models = list(set(r.model_name for r in records))
        
        # Best times per solver type
        best_times = {}
        for solver in solver_types:
            solver_records = [r for r in records if r.solver_type == solver and r.converged]
            if solver_records:
                best = min(solver_records, key=lambda r: r.timings.get('total_program_time', float('inf')))
                best_times[solver] = {
                    "total_time": best.timings.get('total_program_time'),
                    "model": best.model_name,
                    "record_id": best.id
                }
        
        return {
            "total_records": len(records),
            "solver_types": solver_types,
            "models": models,
            "best_times": best_times,
            "server_config": self.server_config,
            "server_hash": self.server_hash
        }


# =============================================================================
# FastAPI Router for Benchmark API
# =============================================================================

def create_benchmark_router(service: BenchmarkService):
    """Create FastAPI router for benchmark API endpoints."""
    from fastapi import APIRouter, HTTPException, Query
    
    router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])
    
    @router.get("")
    async def list_benchmarks(
        sort_by: str = Query("timestamp", description="Field to sort by"),
        sort_order: str = Query("desc", regex="^(asc|desc)$"),
        solver_type: Optional[str] = Query(None, description="Filter by solver type"),
        model_name: Optional[str] = Query(None, description="Filter by model name")
    ):
        """List all benchmark records with sorting and filtering."""
        filters = {}
        if solver_type:
            filters["solver_type"] = solver_type
        if model_name:
            filters["model_name"] = model_name
        
        records = service.get_all_records(
            sort_by=sort_by,
            sort_order=sort_order,
            filters=filters if filters else None
        )
        
        return {
            "count": len(records),
            "records": records
        }
    
    @router.get("/summary")
    async def get_summary():
        """Get benchmark summary statistics."""
        return service.get_summary()
    
    @router.get("/server-config")
    async def get_server_config():
        """Get current server hardware configuration."""
        return {
            "config": service.server_config,
            "hash": service.server_hash
        }
    
    @router.get("/{record_id}")
    async def get_benchmark(record_id: str):
        """Get a specific benchmark record."""
        record = service.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        return record.to_dict()
    
    @router.delete("/{record_id}")
    async def delete_benchmark(record_id: str):
        """Delete a specific benchmark record."""
        if service.delete_record(record_id):
            return {"status": "deleted", "id": record_id}
        raise HTTPException(status_code=404, detail="Record not found")
    
    @router.post("/refresh")
    async def refresh_data():
        """Force refresh data from file."""
        service.refresh()
        return {"status": "refreshed", "count": len(service._records)}
    
    return router


# =============================================================================
# Socket.IO Event Handler Integration
# =============================================================================

class BenchmarkEventHandler:
    """
    Handles Socket.IO events for benchmark recording.
    
    Integrates with fem_api_server by subscribing to solve_complete events.
    """
    
    def __init__(self, service: BenchmarkService, gallery_file: Optional[Path] = None):
        self.service = service
        self.gallery_data: Dict[str, Any] = {}
        
        # Load gallery data for model name lookup
        if gallery_file and gallery_file.exists():
            try:
                with open(gallery_file, 'r') as f:
                    gallery = json.load(f)
                    for mesh in gallery.get('meshes', []):
                        # Key by filename only (without path)
                        filename = Path(mesh.get('file', '')).name
                        self.gallery_data[filename] = mesh
                print(f"[Benchmark] Loaded {len(self.gallery_data)} models from gallery")
            except Exception as e:
                print(f"[Benchmark] Warning: Could not load gallery: {e}")
    
    def get_model_name(self, mesh_file: str) -> str:
        """Get display name for a mesh file from gallery data."""
        filename = Path(mesh_file).name
        if filename in self.gallery_data:
            return self.gallery_data[filename].get('name', filename)
        return filename
    
    def on_solve_complete(
        self,
        job_data: Dict[str, Any],
        client_config: Optional[Dict[str, Any]] = None
    ) -> Optional[BenchmarkRecord]:
        """
        Handle solve_complete event and record benchmark.
        
        Args:
            job_data: Job data including params, results, mesh_info
            client_config: Client hardware configuration
        
        Returns:
            Created BenchmarkRecord or None if recording failed
        """
        try:
            params = job_data.get('params', {})
            results = job_data.get('results', {})
            mesh_info = results.get('mesh_info', job_data.get('mesh_info', {}))
            
            mesh_file = params.get('mesh_file', 'unknown')
            model_name = self.get_model_name(mesh_file)
            
            record = self.service.add_record(
                model_file=Path(mesh_file).name,
                model_name=model_name,
                model_nodes=mesh_info.get('nodes', 0),
                model_elements=mesh_info.get('elements', 0),
                solver_type=params.get('solver_type', 'unknown'),
                converged=results.get('converged', False),
                iterations=results.get('iterations', 0),
                timings=results.get('timing_metrics', {}),
                solution_stats=results.get('solution_stats', {}),
                client_config=client_config
            )
            
            return record
            
        except Exception as e:
            print(f"[Benchmark] Error recording result: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# Module Initialization Helper
# =============================================================================

def init_benchmark_service(
    data_dir: Path | str,
    gallery_file: Optional[Path | str] = None
) -> tuple[BenchmarkService, BenchmarkEventHandler]:
    """
    Initialize benchmark service and event handler.
    
    Args:
        data_dir: Directory for benchmark data file
        gallery_file: Optional path to gallery_files.json
    
    Returns:
        Tuple of (BenchmarkService, BenchmarkEventHandler)
    """
    data_file = Path(data_dir) / "benchmark_results.json"
    service = BenchmarkService(data_file)
    
    gallery_path = Path(gallery_file) if gallery_file else None
    handler = BenchmarkEventHandler(service, gallery_path)
    
    return service, handler


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    # Test hardware detection
    print("\n=== Server Hardware Detection ===")
    config = detect_server_hardware()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test service
    print("\n=== Benchmark Service Test ===")
    service = BenchmarkService(Path("/tmp/benchmark_test.json"))
    
    # Add test record
    record = service.add_record(
        model_file="test_mesh.h5",
        model_name="Test Model (Small)",
        model_nodes=1000,
        model_elements=250,
        solver_type="gpu",
        converged=True,
        iterations=500,
        timings={
            "load_mesh": 0.1,
            "assemble_system": 0.5,
            "solve_system": 2.0,
            "total_program_time": 2.8
        },
        solution_stats={"u_range": [0, 10]},
        client_config={"browser": "Chrome", "os": "Windows"}
    )
    
    print(f"\nCreated record: {record.id}")
    
    # List records
    records = service.get_all_records(sort_by="total_time", sort_order="asc")
    print(f"\nTotal records: {len(records)}")
    
    # Summary
    summary = service.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
