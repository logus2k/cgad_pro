"""
Result Recorder for Automated Benchmark Suite.

Records benchmark results in the same format as manual testing,
ensuring compatibility with existing UI and reporting tools.

File naming convention: benchmark_{server_hash}_automated.json
"""

import json
import uuid
import hashlib
import platform
import subprocess
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


def detect_cpu_model() -> str:
    """Detect CPU model with cross-platform support."""
    system = platform.system()
    
    if system == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
    
    elif system == "Windows":
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                if len(lines) >= 2:
                    return lines[1]
        except Exception:
            pass
    
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
    
    return platform.processor() or "Unknown"


def detect_os_info() -> str:
    """Detect OS/distribution name with cross-platform support."""
    system = platform.system()
    
    if system == "Linux":
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=', 1)[1].strip().strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"
    
    elif system == "Windows":
        return f"Windows {platform.release()} ({platform.version()})"
    
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    
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


def generate_server_hash(config: Dict[str, Any]) -> str:
    """Generate server hash matching existing benchmark_service.py logic."""
    hash_keys = ["hostname", "cpu_model", "cpu_cores", "ram_gb", "gpu_model", "gpu_memory_gb"]
    hash_input = "|".join(str(config.get(k, "")) for k in sorted(hash_keys))
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


@dataclass
class BenchmarkRecord:
    """Single benchmark result record matching manual testing format."""
    id: str
    timestamp: str
    model_file: str
    model_name: str
    model_nodes: int
    model_elements: int
    solver_type: str
    converged: bool
    iterations: int
    timings: Dict[str, float]
    solution_stats: Dict[str, Any]
    memory: Dict[str, Any]
    server_config: Dict[str, Any]
    server_hash: str
    client_config: Dict[str, Any]
    client_hash: str
    matrix_nnz: int = 0
    element_type: str = "quad8"
    nodes_per_element: int = 8
    solver_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.solver_config is None:
            self.solver_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultRecorder:
    """
    Records benchmark results to JSON file.
    
    Maintains compatibility with existing benchmark_service.py format.
    File naming: benchmark_{server_hash}_automated.json
    """
    
    VERSION = "1.1"
    
    def __init__(self, data_dir: Path):
        """
        Initialize result recorder.
        
        Args:
            data_dir: Directory for benchmark data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.server_config = detect_server_hardware()
        self.server_hash = generate_server_hash(self.server_config)
        
        # Automated benchmark file
        self.data_file = self.data_dir / f"benchmark_{self.server_hash}_automated.json"
        
        self.records: List[BenchmarkRecord] = []
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing records from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                for r in data.get('records', []):
                    if 'memory' not in r:
                        r['memory'] = {}
                    if 'matrix_nnz' not in r:
                        r['matrix_nnz'] = 0
                    if 'solver_config' not in r:
                        r['solver_config'] = {}
                    
                    self.records.append(BenchmarkRecord(**r))
                
                print(f"[Recorder] Loaded {len(self.records)} existing records")
            except Exception as e:
                print(f"[Recorder] Warning: Could not load existing data: {e}")
    
    def _save(self) -> None:
        """Save all records to file."""
        data = {
            "version": self.VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "automated": True,
            "server_hash": self.server_hash,
            "server_config": self.server_config,
            "records": [r.to_dict() for r in self.records]
        }
        
        temp_file = self.data_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.data_file)
    
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
        memory: Dict[str, Any],
        mesh_info: Optional[Dict[str, Any]] = None,
        solver_config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkRecord:
        """Add a new benchmark record."""
        record = BenchmarkRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_file=model_file,
            model_name=model_name,
            model_nodes=model_nodes,
            model_elements=model_elements,
            solver_type=solver_type,
            converged=converged,
            iterations=iterations,
            timings=timings,
            solution_stats=solution_stats,
            memory=memory,
            server_config=self.server_config,
            server_hash=self.server_hash,
            client_config={"source": "automated_benchmark"},
            client_hash="automated",
            matrix_nnz=mesh_info.get('matrix_nnz', 0) if mesh_info else 0,
            element_type=mesh_info.get('element_type', 'quad8') if mesh_info else 'quad8',
            nodes_per_element=mesh_info.get('nodes_per_element', 8) if mesh_info else 8,
            solver_config=solver_config or {}
        )
        
        self.records.append(record)
        self._save()
        
        return record
    
    def clear_records(self) -> int:
        """Clear all automated benchmark records."""
        count = len(self.records)
        self.records = []
        self._save()
        return count
    
    def get_record_count(self) -> int:
        """Get number of records."""
        return len(self.records)
    
    def get_server_display_name(self) -> str:
        """Get display name for the server."""
        hostname = self.server_config.get('hostname', 'UNKNOWN')
        return hostname.upper()
    
    def has_result(self, solver_type: str, model_name: str, mesh_nodes: int) -> bool:
        """
        Check if a result already exists for this test combination.
        Used for --resume functionality.
        """
        for r in self.records:
            if (r.solver_type == solver_type and 
                r.model_name == model_name and 
                r.model_nodes == mesh_nodes):
                return True
        return False
    
    def count_existing_for_test(self, solver_type: str, model_name: str, mesh_nodes: int) -> int:
        """Count how many results exist for this test combination."""
        count = 0
        for r in self.records:
            if (r.solver_type == solver_type and 
                r.model_name == model_name and 
                r.model_nodes == mesh_nodes):
                count += 1
        return count
