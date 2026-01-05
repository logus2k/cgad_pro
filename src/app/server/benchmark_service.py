"""
Benchmark Service - Records and serves FEM solver benchmark results.

This service:
- Subscribes to Socket.IO solver events (solve_complete)
- Auto-detects server hardware configuration
- Records benchmark results with timestamps
- Persists data to server-specific JSON files
- Aggregates records from multiple server files for display
- Provides REST API for benchmark data access

Location: /src/app/server/benchmark_service.py
Data files: /src/app/server/benchmark/benchmark_{server_hash}.json
"""

import os
import json
import uuid
import hashlib
import platform
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import threading
import asyncio


# =============================================================================
# Server Hardware Detection
# =============================================================================

def detect_cpu_model() -> str:
    """Detect CPU model with cross-platform support."""
    system = platform.system()
    
    # Linux / WSL
    if system == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
    
    # Windows
    elif system == "Windows":
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                if len(lines) >= 2:
                    return lines[1]  # First line is "Name", second is the value
        except Exception:
            pass
    
    # macOS
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    
    # Fallback for any platform
    return platform.processor() or "Unknown"

def detect_os_info() -> str:
    """Detect OS/distribution name with cross-platform support."""
    system = platform.system()
    
    if system == "Linux":
        # Try to get distro info from /etc/os-release
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=', 1)[1].strip().strip('"')
        except Exception:
            pass
        # Fallback to kernel info
        return f"Linux {platform.release()}"
    
    elif system == "Windows":
        # e.g., "Windows 10 (10.0.19045)"
        return f"Windows {platform.release()} ({platform.version()})"
    
    elif system == "Darwin":
        # macOS
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    
    else:
        # Generic fallback for other systems
        return f"{system} {platform.release()}"

def detect_server_hardware() -> Dict[str, Any]:
    """Detect server hardware configuration."""
    config = {
        "hostname": platform.node(),
        "os": detect_os_info(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": detect_cpu_model(),
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
    # Required fields (no defaults) - must come first
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
    
    # Fields with defaults - must come after required fields
    matrix_nnz: int = 0
    element_type: str = "quad8"
    nodes_per_element: int = 8
    solver_config: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    server_config: Dict[str, Any] = field(default_factory=dict)
    server_hash: str = ""
    client_config: Dict[str, Any] = field(default_factory=dict)
    client_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkRecord':
        # Handle records without new fields (backward compatibility)
        if 'memory' not in data:
            data['memory'] = {}
        if 'matrix_nnz' not in data:
            data['matrix_nnz'] = 0
        if 'element_type' not in data:
            data['element_type'] = 'quad8'
        if 'nodes_per_element' not in data:
            data['nodes_per_element'] = 8
        if 'solver_config' not in data:
            data['solver_config'] = {}
        return cls(**data)


# =============================================================================
# Benchmark Service
# =============================================================================

class BenchmarkService:
    """
    Service for recording and serving benchmark results.
    
    Features:
    - In-memory cache for fast access
    - Server-specific JSON file persistence
    - Aggregates records from multiple server files
    - Auto-reload on file changes
    - Thread-safe operations
    """
    
    # Keys used for server config hashing
    SERVER_HASH_KEYS = ["hostname", "cpu_model", "cpu_cores", "ram_gb", "gpu_model", "gpu_memory_gb"]
    
    # Keys used for client config hashing
    CLIENT_HASH_KEYS = ["browser", "os", "gpu_vendor", "gpu_renderer"]
    
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._records: Dict[str, BenchmarkRecord] = {}
        self._file_mtimes: Dict[str, float] = {}  # Track mtime per file
        
        # Detect server hardware once at startup
        self.server_config = detect_server_hardware()
        self.server_hash = generate_config_hash(self.server_config, self.SERVER_HASH_KEYS)
        
        # This server's data file
        self.own_data_file = self.data_dir / f"benchmark_{self.server_hash}.json"
        
        print(f"[Benchmark] Server config hash: {self.server_hash}")
        print(f"[Benchmark] Data file: {self.own_data_file.name}")
        print(f"[Benchmark] CPU: {self.server_config['cpu_model']} ({self.server_config['cpu_cores']} cores)")
        print(f"[Benchmark] RAM: {self.server_config['ram_gb']} GB")
        if self.server_config['gpu_model']:
            print(f"[Benchmark] GPU: {self.server_config['gpu_model']} ({self.server_config['gpu_memory_gb']} GB)")
        
        # Migrate legacy file if exists
        self._migrate_legacy_file()
        
        # Load existing data from all files
        self._load_all_files()
    
    def _migrate_legacy_file(self) -> None:
        """Migrate legacy benchmark_results.json to server-specific file."""
        legacy_file = self.data_dir / "benchmark_results.json"
        
        if legacy_file.exists() and not self.own_data_file.exists():
            try:
                # Rename legacy file to this server's file
                legacy_file.rename(self.own_data_file)
                print(f"[Benchmark] Migrated legacy file to {self.own_data_file.name}")
            except Exception as e:
                print(f"[Benchmark] Warning: Could not migrate legacy file: {e}")
    
    def _load_all_files(self) -> None:
        """Load benchmark data from all JSON files in the data directory."""
        with self._lock:
            self._records = {}
            
            # Find all benchmark_*.json files
            json_files = list(self.data_dir.glob("benchmark_*.json"))
            
            if not json_files:
                print(f"[Benchmark] No benchmark files found in {self.data_dir}")
                return
            
            total_loaded = 0
            
            for json_file in json_files:
                try:
                    mtime = json_file.stat().st_mtime
                    
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    records = data.get('records', [])
                    for r in records:
                        record = BenchmarkRecord.from_dict(r)
                        self._records[record.id] = record
                    
                    self._file_mtimes[str(json_file)] = mtime
                    total_loaded += len(records)
                    
                    print(f"[Benchmark] Loaded {len(records)} records from {json_file.name}")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Benchmark] Warning: Could not load {json_file.name}: {e}")
                except Exception as e:
                    print(f"[Benchmark] Warning: Error reading {json_file.name}: {e}")
            
            print(f"[Benchmark] Total: {total_loaded} records from {len(json_files)} file(s)")
    
    def _save_to_file(self) -> None:
        """Save this server's records to appropriate JSON files (manual and/or automated)."""
        with self._lock:
            # Separate records by type: manual vs automated
            manual_records = []
            automated_records = []
            
            for r in self._records.values():
                if r.server_hash != self.server_hash:
                    continue  # Skip records from other servers
                
                if r.client_hash == "automated":
                    automated_records.append(r.to_dict())
                else:
                    manual_records.append(r.to_dict())
            
            # Save manual records to own_data_file
            self._write_records_file(self.own_data_file, manual_records)
            
            # Save automated records to _automated file
            automated_file = self.data_dir / f"benchmark_{self.server_hash}_automated.json"
            self._write_records_file(automated_file, automated_records)
    
    def _write_records_file(self, file_path: Path, records: list) -> None:
        """Write records to a JSON file atomically."""
        if not records:
            # If no records, delete the file if it exists
            if file_path.exists():
                file_path.unlink()
                self._file_mtimes.pop(str(file_path), None)
                print(f"[Benchmark] Removed empty file: {file_path.name}")
            return
        
        data = {
            "version": "1.1",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "records": records
        }
        
        # Write atomically
        temp_file = file_path.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(file_path)
        
        self._file_mtimes[str(file_path)] = file_path.stat().st_mtime
    
    def refresh(self) -> None:
        """Refresh data from all files if any changed externally."""
        self._load_all_files()
    
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
        matrix_nnz: int = 0,
        element_type: str = "quad8",
        nodes_per_element: int = 8,
        solver_config: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
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
            memory: Memory usage statistics (peak_ram_mb, peak_vram_mb, etc.)
            client_config: Client hardware config (from browser)
        
        Returns:
            Created BenchmarkRecord
        """
        record_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"        
        client_config = client_config or {}
        memory = memory or {}
        client_hash = generate_config_hash(client_config, self.CLIENT_HASH_KEYS) if client_config else "unknown"
        
        record = BenchmarkRecord(
            id=record_id,
            timestamp=timestamp,
            model_file=model_file,
            model_name=model_name,
            model_nodes=model_nodes,
            model_elements=model_elements,
            solver_type=solver_type,
            converged=converged,
            iterations=iterations,
            timings=timings,
            solution_stats=solution_stats,
            matrix_nnz=matrix_nnz,
            element_type=element_type,
            nodes_per_element=nodes_per_element,
            solver_config=solver_config or {},
            memory=memory or {},
            server_config=self.server_config,
            server_hash=self.server_hash,
            client_config=client_config or {},
            client_hash=client_hash
        )
        
        with self._lock:
            self._records[record.id] = record
            self._save_to_file()
        
        # Log with memory info if available
        mem_info = ""
        if memory:
            peak_ram = memory.get('peak_ram_mb', 0)
            peak_vram = memory.get('peak_vram_mb', 0)
            mem_info = f", RAM: {peak_ram:.0f}MB, VRAM: {peak_vram:.0f}MB"
        
        print(f"[Benchmark] Added record {record.id[:8]}... "
              f"({model_name}, {solver_type}, {timings.get('total_program_time', 0):.2f}s{mem_info})")
        
        return record
    
    def get_record(self, record_id: str) -> Optional[BenchmarkRecord]:
        """Get a single record by ID."""
        with self._lock:
            return self._records.get(record_id)
    
    def delete_record(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Note: Can only delete records belonging to this server.
        Records from other servers are read-only.
        """
        with self._lock:
            record = self._records.get(record_id)
            if not record:
                return False
            
            # Check if this record belongs to this server
            if record.server_hash != self.server_hash:
                print(f"[Benchmark] Cannot delete record {record_id[:8]}... - belongs to different server")
                return False
            
            del self._records[record_id]
            self._save_to_file()
            return True
    
    def delete_records(self, record_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple records by ID (bulk operation).
        
        Note: Can only delete records belonging to this server.
        Records from other servers are skipped.
        
        Returns:
            Dict with 'deleted' count and 'skipped' count
        """
        deleted = 0
        skipped = 0
        
        with self._lock:
            for record_id in record_ids:
                record = self._records.get(record_id)
                if not record:
                    skipped += 1
                    continue
                
                # Check if this record belongs to this server
                if record.server_hash != self.server_hash:
                    skipped += 1
                    continue
                
                del self._records[record_id]
                deleted += 1
            
            # Save once after all deletions
            if deleted > 0:
                self._save_to_file()
        
        print(f"[Benchmark] Bulk delete: {deleted} deleted, {skipped} skipped")
        return {"deleted": deleted, "skipped": skipped}
    
    def get_all_records(
        self,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all records with optional sorting and filtering.
        
        Args:
            sort_by: Field to sort by (timestamp, total_time, iterations, etc.)
            sort_order: 'asc' or 'desc'
            filters: Optional dict of field:value filters
        
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
        reverse = sort_order == "desc"
        
        def get_sort_key(r: Dict) -> Any:
            if sort_by == "total_time":
                return r.get('timings', {}).get('total_program_time', 0)
            elif sort_by == "peak_ram":
                return r.get('memory', {}).get('peak_ram_mb', 0)
            elif sort_by == "peak_vram":
                return r.get('memory', {}).get('peak_vram_mb', 0)
            return r.get(sort_by, '')
        
        records.sort(key=get_sort_key, reverse=reverse)
        
        return records
    
    def get_records_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all records for a specific model."""
        return self.get_all_records(filters={"model_name": model_name})
    
    def get_records_by_solver(self, solver_type: str) -> List[Dict[str, Any]]:
        """Get all records for a specific solver type."""
        return self.get_all_records(filters={"solver_type": solver_type})
    
    def get_best_time(self, model_name: str, solver_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the record with best (lowest) total time for a model."""
        filters = {"model_name": model_name}
        if solver_type:
            filters["solver_type"] = solver_type
        
        records = self.get_all_records(sort_by="total_time", sort_order="asc", filters=filters)
        return records[0] if records else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all benchmarks."""
        records = self.get_all_records()
        
        if not records:
            return {
                "total_records": 0,
                "solver_types": [],
                "models": [],
                "best_times": {}
            }
        
        # Collect unique values
        solver_types = list(set(r['solver_type'] for r in records))
        models = list(set(r['model_name'] for r in records))
        
        # Get best times per model
        best_times = {}
        for model in models:
            model_records = [r for r in records if r['model_name'] == model]
            if model_records:
                best = min(model_records, key=lambda r: r.get('timings', {}).get('total_program_time', float('inf')))
                best_times[model] = {
                    "time": best.get('timings', {}).get('total_program_time', 0),
                    "solver": best['solver_type'],
                    "memory": best.get('memory', {})
                }
        
        return {
            "total_records": len(records),
            "solver_types": solver_types,
            "models": models,
            "best_times": best_times,
            "server_config": self.server_config,
            "server_hash": self.server_hash
        }
    
    def get_file_info(self) -> List[Dict[str, Any]]:
        """Get information about all benchmark files in the directory."""
        files = []
        for json_file in self.data_dir.glob("benchmark_*.json"):
            try:
                stat = json_file.stat()
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Detect automated files and extract server hash
                # Format: benchmark_{hash}.json or benchmark_{hash}_automated.json
                basename = json_file.stem.replace("benchmark_", "")
                is_automated = basename.endswith("_automated")
                
                if is_automated:
                    file_hash = basename.replace("_automated", "")
                else:
                    file_hash = basename
                
                # Get hostname from file's server_config for display name
                server_config = data.get('server_config', {})
                hostname = server_config.get('hostname', file_hash).upper()
                
                # Build display name
                display_name = f"{hostname} (Automated)" if is_automated else hostname
                
                files.append({
                    "filename": json_file.name,
                    "server_hash": file_hash,
                    "display_name": display_name,
                    "is_automated": is_automated,
                    "is_own": file_hash == self.server_hash,
                    "record_count": len(data.get('records', [])),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "updated_at": data.get('updated_at', ''),
                    "version": data.get('version', 'unknown')
                })
            except Exception as e:
                files.append({
                    "filename": json_file.name,
                    "error": str(e)
                })
        
        # Sort by display_name for consistent ordering
        files.sort(key=lambda x: x.get('display_name', x.get('filename', '')))
        
        return files


# =============================================================================
# FastAPI Router for Benchmark API
# =============================================================================

def create_benchmark_router(service: BenchmarkService, gallery_file: Optional[Path] = None):
    """Create FastAPI router for benchmark API endpoints."""

    from report_generator import REPORT_SECTIONS

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
    
    @router.get("/report/sections")
    async def get_report_sections():
        """Get list of available report sections."""
        return {"sections": REPORT_SECTIONS}
    
    @router.get("/report/{section_id}")
    async def get_report_section(
        section_id: str,
        solver_type: Optional[str] = Query(None, description="Filter by solver type"),
        model_name: Optional[str] = Query(None, description="Filter by model name"),
        server_hash: Optional[str] = Query(None, description="Filter by server hash (supports _automated suffix)"),
        is_automated: Optional[bool] = Query(None, description="Filter by automated (true) or manual (false) testing")
    ):
        """Generate a specific report section with optional filters."""
        from report_generator import create_report_generator_from_records, REPORT_SECTIONS
        
        # Validate section ID
        valid_ids = [s["id"] for s in REPORT_SECTIONS]
        if section_id not in valid_ids:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid section ID. Valid options: {valid_ids}"
            )
        
        # Build filters dict
        filters = {}
        if solver_type:
            filters["solver_type"] = solver_type
        if model_name:
            filters["model_name"] = model_name
        
        # Handle composite server_hash (e.g., "abc123_automated")
        # Extract actual hash and automated flag from composite key
        is_automated_from_hash = None
        if server_hash:
            if server_hash.endswith("_automated"):
                filters["server_hash"] = server_hash[:-10]  # Remove "_automated" suffix
                is_automated_from_hash = True
            else:
                filters["server_hash"] = server_hash
                is_automated_from_hash = False
        
        # Determine final automated filter:
        # - Explicit is_automated parameter takes precedence
        # - Otherwise use value derived from composite server_hash
        is_automated_filter = is_automated if is_automated is not None else is_automated_from_hash
        
        # Get filtered records as dicts
        filtered_records = service.get_all_records(
            filters=filters if filters else None
        )
        
        # Apply automated filter (client_hash == "automated" for automated records)
        if is_automated_filter is not None:
            if is_automated_filter:
                filtered_records = [r for r in filtered_records if r.get("client_hash") == "automated"]
            else:
                filtered_records = [r for r in filtered_records if r.get("client_hash") != "automated"]
        
        # Create generator with filtered records
        generator = create_report_generator_from_records(
            records=filtered_records,
            server_config=service.server_config,
            gallery_file=gallery_file
        )
        
        result = generator.generate_section(section_id)
        return result
    
    class BulkDeleteRequest(BaseModel):
        ids: List[str]
    
    @router.post("/bulk-delete")
    async def bulk_delete_benchmarks(request: BulkDeleteRequest):
        """Delete multiple benchmark records in one operation."""
        result = service.delete_records(request.ids)
        return {
            "status": "deleted",
            "deleted": result["deleted"],
            "skipped": result["skipped"]
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
        if gallery_file:
            print(f"[Benchmark] Gallery file path: {gallery_file}")
            print(f"[Benchmark] Gallery file exists: {gallery_file.exists()}")
        
        if gallery_file and gallery_file.exists():
            try:
                with open(gallery_file, 'r') as f:
                    gallery = json.load(f)
                    # Iterate through models and their meshes
                    for model in gallery.get('models', []):
                        model_name = model.get('name', '')
                        for mesh in model.get('meshes', []):
                            # Key by filename only (without path)
                            file_path = mesh.get('file', '')
                            filename = Path(file_path).name
                            # Store model name with mesh entry
                            self.gallery_data[filename] = {
                                'name': model_name,
                                'mesh': mesh,
                                'model': model
                            }
                            print(f"[Benchmark] Mapped '{filename}' -> '{model_name}'")
                    print(f"[Benchmark] Loaded {len(self.gallery_data)} meshes from gallery")
            except Exception as e:
                print(f"[Benchmark] Warning: Could not load gallery: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[Benchmark] Warning: Gallery file not found or not provided")
    
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
        """Handle solve_complete event and record benchmark."""
        try:
            params = job_data.get('params', {})
            results = job_data.get('results', {})
            mesh_info = results.get('mesh_info', job_data.get('mesh_info', {}))
            
            mesh_file = params.get('mesh_file', 'unknown')
            model_name = self.get_model_name(mesh_file)
            
            # Extract memory data from results
            memory = results.get('memory', {})
            
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
                matrix_nnz=mesh_info.get('matrix_nnz', 0),
                element_type=mesh_info.get('element_type', 'quad8'),
                nodes_per_element=mesh_info.get('nodes_per_element', 8),
                solver_config=results.get('solver_config', {}),
                memory=memory,
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
        data_dir: Directory for benchmark data files
        gallery_file: Optional path to gallery_files.json
    
    Returns:
        Tuple of (BenchmarkService, BenchmarkEventHandler)
    """
    service = BenchmarkService(data_dir)
    
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
    test_dir = Path("/tmp/benchmark_test")
    test_dir.mkdir(exist_ok=True)
    
    service = BenchmarkService(test_dir)
    
    # Add test record with memory data
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
        memory={
            "peak_ram_mb": 512.5,
            "peak_vram_mb": 1024.0,
            "baseline_ram_mb": 256.0,
            "baseline_vram_mb": 0.0,
            "sample_count": 28,
            "interval_ms": 100,
            "gpu_available": True
        },
        client_config={"browser": "Chrome", "os": "Windows"}
    )
    
    print(f"\nCreated record: {record.id}")
    print(f"Data file: {service.own_data_file}")
    
    # List files
    print("\n=== Benchmark Files ===")
    for f in service.get_file_info():
        print(f"  {f['filename']}: {f.get('record_count', '?')} records, {f.get('size_kb', '?')} KB")
    
    # List records
    records = service.get_all_records(sort_by="total_time", sort_order="asc")
    print(f"\nTotal records: {len(records)}")
    
    # Summary
    summary = service.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
