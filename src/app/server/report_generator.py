"""
Report Generator - Generates Markdown reports from benchmark data.

Produces sections of the FEM Solver Performance Benchmark Report.

Location: /src/app/server/report_generator.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# =============================================================================
# Report Section Definitions
# =============================================================================

REPORT_SECTIONS = [
    {"id": "executive_summary", "title": "Mesh Performance"},
    {"id": "test_configuration", "title": "Testing Environment"},
    {"id": "results_total_time", "title": "Timing Totals"},
    {"id": "results_stage_breakdown", "title": "Stage Breakdown"},
    {"id": "scaling_analysis", "title": "Scaling Analysis"},
    {"id": "convergence", "title": "Convergence Verification"},
    {"id": "efficiency_metrics", "title": "Efficiency Metrics"},
    {"id": "reproducibility", "title": "Testing Reproducibility"},
    {"id": "analysis", "title": "Critical Analysis"},
    {"id": "conclusions", "title": "Conclusions"},
]

# Solver display order and metadata
SOLVER_ORDER = ["cpu", "cpu_threaded", "cpu_multiprocess", "numba", "numba_cuda", "gpu"]

SOLVER_NAMES = {
    "cpu": "CPU Baseline",
    "cpu_threaded": "CPU Threaded",
    "cpu_multiprocess": "CPU Multiprocess",
    "numba": "Numba CPU",
    "numba_cuda": "Numba CUDA",
    "gpu": "CuPy GPU"
}

SOLVER_FILES = {
    "cpu": "quad8_cpu_v3.py",
    "cpu_threaded": "quad8_cpu_threaded.py",
    "cpu_multiprocess": "quad8_cpu_multiprocess.py",
    "numba": "quad8_numba.py",
    "numba_cuda": "quad8_numba_cuda.py",
    "gpu": "quad8_gpu_v3.py"
}

SOLVER_PARALLELISM = {
    "cpu": "Sequential Python loops",
    "cpu_threaded": "ThreadPoolExecutor (GIL-limited)",
    "cpu_multiprocess": "multiprocessing.Pool",
    "numba": "@njit + prange",
    "numba_cuda": "@cuda.jit kernels",
    "gpu": "CUDA C RawKernels"
}


# =============================================================================
# Helper Functions
# =============================================================================

def format_time(seconds: Optional[float]) -> str:
    """Format time value for display."""
    if seconds is None:
        return "-"
    if seconds < 0.01:
        return "<0.01s"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def format_number(num: Optional[int]) -> str:
    """Format number with thousand separators."""
    if num is None:
        return "-"
    return f"{num:,}"


def format_speedup(speedup: Optional[float]) -> str:
    """Format speedup ratio."""
    if speedup is None:
        return "-"
    return f"{speedup:.1f}x"


def format_percentage(value: Optional[float]) -> str:
    """Format percentage value."""
    if value is None:
        return "-"
    return f"{value:.1f}%"


def format_memory(mb: Optional[float]) -> str:
    """Format memory in MB to human-readable string."""
    if mb is None or mb == 0:
        return "-"
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.2f} GB"


def calculate_speedup(baseline: float, optimized: float) -> Optional[float]:
    """Calculate speedup ratio."""
    if baseline is None or optimized is None or optimized == 0:
        return None
    return baseline / optimized


# =============================================================================
# Statistical Helper Functions (for multi-server aggregation)
# =============================================================================

def compute_mean(values: List[float]) -> Optional[float]:
    """Compute mean of a list of values."""
    if not values:
        return None
    return sum(values) / len(values)


def compute_std(values: List[float]) -> Optional[float]:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def format_time_with_std(mean: Optional[float], std: Optional[float], n: int = 0) -> str:
    """Format time with standard deviation."""
    if mean is None:
        return "-"
    
    mean_str = format_time(mean)
    
    if std is not None and n > 1:
        # Format std in same units as mean for consistency
        if mean < 1:
            std_str = f"{std * 1000:.0f}ms"
        elif mean < 60:
            std_str = f"{std:.2f}s"
        else:
            std_str = f"{std:.1f}s"
        return f"{mean_str} ± {std_str}"
    
    return mean_str


def format_percentage_with_std(mean: Optional[float], std: Optional[float], n: int = 0) -> str:
    """Format percentage with standard deviation."""
    if mean is None:
        return "-"
    if std is not None and n > 1:
        return f"{mean:.1f}% ± {std:.1f}%"
    return f"{mean:.1f}%"


def get_timing_stats(records: List[Dict], timing_key: str) -> Dict[str, Any]:
    """
    Compute statistics for a specific timing across multiple records.
    
    Returns:
        Dict with 'mean', 'std', 'n', 'values'
    """
    values = []
    for rec in records:
        val = rec.get("timings", {}).get(timing_key)
        if val is not None:
            values.append(val)
    
    return {
        "mean": compute_mean(values),
        "std": compute_std(values),
        "n": len(values),
        "values": values
    }


def get_iteration_stats(records: List[Dict]) -> Dict[str, Any]:
    """
    Get iteration statistics and check for determinism.
    
    Returns:
        Dict with 'value', 'is_deterministic', 'values', 'n'
    """
    values = []
    for rec in records:
        val = rec.get("iterations")
        if val is not None:
            values.append(val)
    
    if not values:
        return {"value": None, "is_deterministic": True, "values": [], "n": 0}
    
    # Check if all values are the same (deterministic)
    is_deterministic = len(set(values)) == 1
    
    return {
        "value": values[0] if is_deterministic else compute_mean(values),
        "is_deterministic": is_deterministic,
        "values": values,
        "n": len(values)
    }


def get_residual_stats(records: List[Dict], key: str) -> Dict[str, Any]:
    """
    Get residual statistics and check for determinism.
    
    Args:
        records: List of record dicts
        key: 'final_residual' or 'relative_residual'
    
    Returns:
        Dict with 'value', 'is_deterministic', 'n'
    """
    values = []
    for rec in records:
        val = rec.get("solution_stats", {}).get(key)
        if val is not None:
            values.append(val)
    
    if not values:
        return {"value": None, "is_deterministic": True, "n": 0}
    
    # Check determinism with tolerance (floating point)
    if len(values) > 1:
        mean_val = sum(values) / len(values)
        # Allow 1% tolerance for floating point
        is_deterministic = all(abs(v - mean_val) / (mean_val + 1e-15) < 0.01 for v in values)
    else:
        is_deterministic = True
    
    return {
        "value": values[0] if is_deterministic else compute_mean(values),
        "is_deterministic": is_deterministic,
        "n": len(values)
    }


def collect_unique_servers(records: List[Dict]) -> List[Dict[str, Any]]:
    """
    Collect unique server configurations from records.
    
    Returns:
        List of server config dicts with record counts
    """
    servers = {}
    for rec in records:
        server_hash = rec.get("server_hash", "unknown")
        if server_hash not in servers:
            config = rec.get("server_config", {})
            servers[server_hash] = {
                "hash": server_hash,
                "hostname": config.get("hostname", "Unknown"),
                "cpu_model": config.get("cpu_model", "Unknown"),
                "cpu_cores": config.get("cpu_cores", "?"),
                "ram_gb": config.get("ram_gb", "?"),
                "gpu_model": config.get("gpu_model"),
                "gpu_memory_gb": config.get("gpu_memory_gb"),
                "os": config.get("os", "Unknown"),
                "record_count": 0
            }
        servers[server_hash]["record_count"] += 1
    
    return list(servers.values())


def get_best_record_per_solver(records: List[Dict], model_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Get the best (fastest) record for each solver type.
    Optionally filter by model name.
    """
    best = {}
    for record in records:
        if model_name and record.get("model_name") != model_name:
            continue
        
        solver = record.get("solver_type")
        if not solver:
            continue
        
        total_time = record.get("timings", {}).get("total_program_time")
        if total_time is None:
            continue
        
        if solver not in best or total_time < best[solver].get("timings", {}).get("total_program_time", float("inf")):
            best[solver] = record
    
    return best


def group_records_by_model(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by model name."""
    groups = {}
    for record in records:
        model = record.get("model_name", "Unknown")
        if model not in groups:
            groups[model] = []
        groups[model].append(record)
    return groups


def group_records_by_model_and_size(records: List[Dict]) -> Dict[tuple, List[Dict]]:
    """
    Group records by (model_name, model_nodes) combination.
    Returns dict with (model_name, nodes) tuples as keys.
    """
    groups = {}
    for record in records:
        model = record.get("model_name", "Unknown")
        nodes = record.get("model_nodes", 0)
        key = (model, nodes)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    return groups


def get_sorted_model_size_keys(groups: Dict[tuple, List[Dict]]) -> List[tuple]:
    """
    Sort (model_name, nodes) keys by model name then by node count.
    """
    return sorted(groups.keys(), key=lambda x: (x[0], x[1]))


def get_model_size_label(nodes: int) -> str:
    """Get size label based on node count."""
    if nodes < 1000:
        return "XS"
    elif nodes < 100000:
        return "S"
    elif nodes < 500000:
        return "M"
    elif nodes < 1000000:
        return "L"
    else:
        return "XL"


# =============================================================================
# Report Generator Class
# =============================================================================

class ReportGenerator:
    """
    Generates Markdown report sections from benchmark data.
    """
    
    def __init__(
        self,
        records: List[Dict[str, Any]],
        server_config: Dict[str, Any],
        gallery_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize report generator.
        
        Args:
            records: List of benchmark records
            server_config: Server hardware configuration
            gallery_data: Optional gallery data for model metadata
        """
        self.records = records
        self.server_config = server_config
        self.gallery_data = gallery_data or {}
        
        # Pre-process data
        self.records_by_model = group_records_by_model(records)
        self.models = list(self.records_by_model.keys())
        
        # Group by model + mesh size (for detailed reports)
        self.records_by_model_size = group_records_by_model_and_size(records)
        self.model_size_keys = get_sorted_model_size_keys(self.records_by_model_size)
        
        # Get unique solvers present in data
        self.solvers_present = set()
        for record in records:
            solver = record.get("solver_type")
            if solver:
                self.solvers_present.add(solver)
        
        # Order solvers according to predefined order
        self.solvers = [s for s in SOLVER_ORDER if s in self.solvers_present]
        
        # Detect if records come from multiple servers
        self.server_hashes = set()
        self.server_hostnames = {}
        for record in records:
            sh = record.get("server_hash")
            if sh:
                self.server_hashes.add(sh)
                # Track hostname for each server hash
                rec_config = record.get("server_config", {})
                if sh not in self.server_hostnames and rec_config.get("hostname"):
                    self.server_hostnames[sh] = rec_config.get("hostname")
        
        self.multi_server = len(self.server_hashes) > 1
    
    def get_available_sections(self) -> List[Dict[str, str]]:
        """Return list of available report sections."""
        return REPORT_SECTIONS
    
    def generate_section(self, section_id: str) -> Dict[str, str]:
        """
        Generate a specific report section.
        
        Args:
            section_id: Section identifier
            
        Returns:
            Dict with 'section', 'title', and 'markdown' keys
        """
        generators = {
            "executive_summary": self._generate_executive_summary,
            "test_configuration": self._generate_test_configuration,
            "results_total_time": self._generate_results_total_time,
            "results_stage_breakdown": self._generate_results_stage_breakdown,
            "scaling_analysis": self._generate_scaling_analysis,
            "convergence": self._generate_convergence,
            "efficiency_metrics": self._generate_efficiency_metrics,
            "analysis": self._generate_analysis,
            "conclusions": self._generate_conclusions,
            "reproducibility": self._generate_reproducibility,
        }
        
        generator = generators.get(section_id)
        if not generator:
            return {
                "section": section_id,
                "title": "Unknown Section",
                "markdown": f"**Error:** Unknown section '{section_id}'"
            }
        
        # Find section title
        title = section_id
        for sec in REPORT_SECTIONS:
            if sec["id"] == section_id:
                title = sec["title"]
                break
        
        markdown = generator()
        
        return {
            "section": section_id,
            "title": title,
            "markdown": markdown
        }
    
    # =========================================================================
    # Section Generators
    # =========================================================================
    
    def _get_multi_server_warning(self) -> str:
        """Generate warning message if data comes from multiple servers."""
        if not self.multi_server:
            return ""
        
        hostnames = list(self.server_hostnames.values())
        if hostnames:
            servers_str = ", ".join(hostnames[:5])
            if len(hostnames) > 5:
                servers_str += f" (+{len(hostnames) - 5} more)"
        else:
            servers_str = f"{len(self.server_hashes)} servers"
        
        return f"\n> **Note:** This report aggregates data from {len(self.server_hashes)} servers ({servers_str}). " \
               f"Values shown are mean ± standard deviation across all runs.\n"
    
    def _get_solver_stats_for_size(self, model_name: str, nodes: int) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated statistics for all solvers for a given model+size.
        
        Returns:
            Dict mapping solver_type to stats dict with:
                - total_time: {mean, std, n}
                - stages: {stage_name: {mean, std, n}, ...}
                - iterations: {value, is_deterministic, n}
                - residuals: {final: {...}, relative: {...}}
                - memory: {peak_ram: {...}, peak_vram: {...}}
                - records: list of all records
        """
        size_records = self.records_by_model_size.get((model_name, nodes), [])
        
        # Group by solver
        by_solver = {}
        for rec in size_records:
            solver = rec.get("solver_type")
            if solver:
                if solver not in by_solver:
                    by_solver[solver] = []
                by_solver[solver].append(rec)
        
        result = {}
        for solver, recs in by_solver.items():
            # Total time stats
            total_stats = get_timing_stats(recs, "total_program_time")
            
            # Stage timing stats
            stage_names = ["load_mesh", "assemble_system", "apply_bc", "solve_system", "compute_derived"]
            stages = {stage: get_timing_stats(recs, stage) for stage in stage_names}
            
            # Iteration stats (check determinism)
            iter_stats = get_iteration_stats(recs)
            
            # Residual stats
            final_res_stats = get_residual_stats(recs, "final_residual")
            rel_res_stats = get_residual_stats(recs, "relative_residual")
            
            # Memory stats
            ram_values = [r.get("memory", {}).get("peak_ram_mb") for r in recs if r.get("memory", {}).get("peak_ram_mb")]
            vram_values = [r.get("memory", {}).get("peak_vram_mb") for r in recs if r.get("memory", {}).get("peak_vram_mb")]
            
            result[solver] = {
                "total_time": total_stats,
                "stages": stages,
                "iterations": iter_stats,
                "residuals": {
                    "final": final_res_stats,
                    "relative": rel_res_stats
                },
                "memory": {
                    "peak_ram": {"mean": compute_mean(ram_values), "std": compute_std(ram_values), "n": len(ram_values)},
                    "peak_vram": {"mean": compute_mean(vram_values), "std": compute_std(vram_values), "n": len(vram_values)}
                },
                "records": recs,
                "n": len(recs)
            }
        
        return result

    def _generate_executive_summary(self) -> str:
        """Generate Executive Summary (Mesh Performance) section."""
        lines = ["Key results from performance benchmarks comparing FEM solver implementations."]
        
        # Add multi-server note if applicable
        if self.multi_server:
            lines.append("")
            lines.append(f"> **Aggregated Results:** Data from {len(self.server_hashes)} servers. "
                        f"Values shown are mean ± std across all runs.")
        
        if not self.records:
            lines.append("\n*No benchmark data available.*")
            return "\n".join(lines)
        
        # Show results for each model+size combination
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            # Get baseline (CPU) mean time for speedup calculation
            baseline_mean = solver_stats.get("cpu", {}).get("total_time", {}).get("mean")
            size_label = get_model_size_label(nodes)
            
            lines.append("")
            lines.append(f"**{model_name} ({size_label})** ({format_number(nodes)} nodes)")
            lines.append("")
            lines.append("| Implementation | Total Time | Speedup vs Baseline | N |")
            lines.append("|----------------|------------|---------------------|---|")
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats:
                    time_mean = stats["total_time"]["mean"]
                    time_std = stats["total_time"]["std"]
                    n = stats["total_time"]["n"]
                    
                    # Compute speedup from averaged times
                    speedup = calculate_speedup(baseline_mean, time_mean) if baseline_mean else None
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    time_str = format_time_with_std(time_mean, time_std, n)
                    speedup_str = format_speedup(speedup) if solver != "cpu" else "1.0x"
                    
                    lines.append(f"| {name} | {time_str} | {speedup_str} | {n} |")
        
        return "\n".join(lines)
    
    def _generate_test_configuration(self) -> str:
        """Generate Test Configuration section."""
        lines = []
        
        # Collect unique servers from records
        servers = collect_unique_servers(self.records)
        
        if self.multi_server:
            # Multi-server: show all contributing servers
            lines.extend([
                "## Contributing Servers",
                "",
                f"Benchmark data aggregated from **{len(servers)} servers**:",
                "",
                "| # | Hostname | CPU | Cores | RAM | GPU | VRAM | Records |",
                "|---|----------|-----|-------|-----|-----|------|---------|",
            ])
            
            for i, srv in enumerate(sorted(servers, key=lambda x: x["hostname"]), 1):
                hostname = srv["hostname"].upper()
                cpu = srv["cpu_model"]
                # Shorten CPU name for table
                if len(cpu) > 30:
                    cpu = cpu[:27] + "..."
                cores = srv["cpu_cores"]
                ram = f"{srv['ram_gb']} GB" if srv['ram_gb'] else "-"
                gpu = srv["gpu_model"] or "-"
                if len(gpu) > 20:
                    gpu = gpu[:17] + "..."
                vram = f"{srv['gpu_memory_gb']} GB" if srv.get("gpu_memory_gb") else "-"
                records = srv["record_count"]
                
                lines.append(f"| {i} | {hostname} | {cpu} | {cores} | {ram} | {gpu} | {vram} | {records} |")
            
            lines.append("")
        else:
            # Single server: show detailed hardware info
            lines.extend([
                "## Hardware Environment",
                "",
                "| Component | Specification |",
                "|-----------|---------------|",
            ])
            
            config = self.server_config
            lines.append(f"| CPU | {config.get('cpu_model', 'Unknown')} ({config.get('cpu_cores', '?')} cores) |")
            lines.append(f"| RAM | {config.get('ram_gb', '?')} GB |")
            
            if config.get('gpu_model'):
                lines.append(f"| GPU | {config.get('gpu_model')} ({config.get('gpu_memory_gb', '?')} GB VRAM) |")
            
            if config.get('cuda_version'):
                lines.append(f"| CUDA Version | {config.get('cuda_version')} |")
            
            lines.append(f"| OS | {config.get('os', 'Unknown')} |")
            lines.append(f"| Python | {config.get('python_version', 'Unknown')} |")
            lines.append("")
        
        # Test Meshes
        lines.extend([
            "## Test Meshes",
            "",
            "| Model | Size | Nodes | Elements | Matrix NNZ |",
            "|-------|------|-------|----------|------------|",
        ])
        
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if size_records:
                rec = size_records[0]
                size_label = get_model_size_label(nodes)
                elements = format_number(rec.get("model_elements"))
                nnz = format_number(rec.get("matrix_nnz", 0))
                lines.append(f"| {model_name} | {size_label} | {format_number(nodes)} | {elements} | {nnz} |")
        
        # Solver Configuration
        lines.extend([
            "",
            "## Solver Configuration",
            "",
        ])
        
        # Get solver config from first record with it
        solver_config = {}
        for rec in self.records:
            sc = rec.get("solver_config", {})
            if sc:
                solver_config = sc
                break
        
        # Get problem type from gallery
        problem_type = "2D Potential Flow (Laplace)"
        for model_data in self.gallery_data.get("models", []):
            pt = model_data.get("problem_type")
            if pt:
                problem_type = pt
                break
        
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Problem Type | {problem_type} |")
        lines.append(f"| Element Type | Quad-8 (8-node serendipity quadrilateral) |")
        lines.append(f"| Linear Solver | {solver_config.get('linear_solver', 'CG').upper()} |")
        lines.append(f"| Tolerance | {solver_config.get('tolerance', 1e-8):.0e} |")
        lines.append(f"| Max Iterations | {format_number(solver_config.get('max_iterations', 50000))} |")
        lines.append(f"| Preconditioner | {solver_config.get('preconditioner', 'Jacobi').title()} |")
        
        # Implementations
        lines.extend([
            "",
            "## Implementations Tested",
            "",
            "| # | Implementation | File | Parallelism Strategy |",
            "|---|----------------|------|----------------------|",
        ])
        
        for i, solver in enumerate(SOLVER_ORDER, 1):
            name = SOLVER_NAMES.get(solver, solver)
            file = SOLVER_FILES.get(solver, "unknown")
            parallelism = SOLVER_PARALLELISM.get(solver, "Unknown")
            lines.append(f"| {i} | {name} | `{file}` | {parallelism} |")
        
        return "\n".join(lines)
    
    def _generate_results_total_time(self) -> str:
        """Generate Results: Total Time section."""
        lines = [
            "## Total Workflow Time",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        if self.multi_server:
            lines.append(f"> Values are mean ± std across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Generate table for each model+size (cleaner than one giant table)
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            baseline_mean = solver_stats.get("cpu", {}).get("total_time", {}).get("mean")
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "| Implementation | Total Time | Speedup | N |",
                "|----------------|------------|---------|---|",
            ])
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats:
                    time_mean = stats["total_time"]["mean"]
                    time_std = stats["total_time"]["std"]
                    n = stats["total_time"]["n"]
                    
                    # Speedup from averaged times
                    speedup = calculate_speedup(baseline_mean, time_mean)
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    time_str = format_time_with_std(time_mean, time_std, n)
                    speedup_str = format_speedup(speedup) if solver != "cpu" else "1.0x"
                    
                    lines.append(f"| {name} | {time_str} | {speedup_str} | {n} |")
            
            lines.append("")
        
        # Speedup comparison table (compact view across all sizes)
        lines.extend([
            "### Speedup Comparison (All Sizes)",
            "",
        ])
        
        header = "| Implementation |"
        separator = "|----------------|"
        for model_name, nodes in self.model_size_keys:
            size_label = get_model_size_label(nodes)
            short_model = model_name[:6] if len(model_name) > 6 else model_name
            header += f" {short_model} {size_label} |"
            separator += "--------|"
        
        lines.append(header)
        lines.append(separator)
        
        for solver in self.solvers:
            row = f"| {SOLVER_NAMES.get(solver, solver)} |"
            for model_name, nodes in self.model_size_keys:
                solver_stats = self._get_solver_stats_for_size(model_name, nodes)
                baseline_mean = solver_stats.get("cpu", {}).get("total_time", {}).get("mean")
                solver_mean = solver_stats.get(solver, {}).get("total_time", {}).get("mean")
                
                if baseline_mean and solver_mean:
                    speedup = calculate_speedup(baseline_mean, solver_mean)
                    row += f" {format_speedup(speedup)} |"
                else:
                    row += " - |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _generate_results_stage_breakdown(self) -> str:
        """Generate Results: Stage Breakdown section."""
        lines = [
            "## Stage-by-Stage Breakdown",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        stages = ["load_mesh", "assemble_system", "apply_bc", "solve_system", "compute_derived", "total_program_time"]
        stage_labels = {
            "load_mesh": "Load Mesh",
            "assemble_system": "Assembly",
            "apply_bc": "Apply BC",
            "solve_system": "Solve",
            "compute_derived": "Post-Process",
            "total_program_time": "**Total**"
        }
        
        if self.multi_server:
            lines.append(f"> Values are mean ± std across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Generate breakdown for each model+size combination
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Stage Timings",
                "",
            ])
            
            # Stage timing table with mean ± std
            header = "| Stage |"
            separator = "|-------|"
            for solver in self.solvers:
                if solver in solver_stats:
                    header += f" {SOLVER_NAMES.get(solver, solver)} |"
                    separator += "---------------|"
            
            lines.append(header)
            lines.append(separator)
            
            for stage in stages:
                label = stage_labels.get(stage, stage)
                row = f"| {label} |"
                for solver in self.solvers:
                    stats = solver_stats.get(solver)
                    if stats:
                        stage_stats = stats["stages"].get(stage) if stage != "total_program_time" else stats["total_time"]
                        if stage_stats:
                            time_str = format_time_with_std(stage_stats["mean"], stage_stats["std"], stage_stats["n"])
                            row += f" {time_str} |"
                        else:
                            row += " - |"
                lines.append(row)
            
            # Time distribution (percentages)
            lines.extend([
                "",
                "#### Time Distribution (% of Total)",
                "",
            ])
            
            header = "| Implementation |"
            separator = "|----------------|"
            for stage in stages[:-1]:  # Exclude total
                header += f" {stage_labels.get(stage, stage)} |"
                separator += "--------|"
            
            lines.append(header)
            lines.append(separator)
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if not stats:
                    continue
                
                row = f"| {SOLVER_NAMES.get(solver, solver)} |"
                total_mean = stats["total_time"]["mean"] or 1
                
                for stage in stages[:-1]:
                    stage_stats = stats["stages"].get(stage, {})
                    stage_mean = stage_stats.get("mean", 0) or 0
                    pct = (stage_mean / total_mean * 100) if total_mean > 0 else 0
                    row += f" {format_percentage(pct)} |"
                
                lines.append(row)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_scaling_analysis(self) -> str:
        """Generate Scaling Analysis section."""
        lines = [
            "## Scaling Analysis",
            "",
            "### Speedup vs Problem Size",
            "",
            "How speedup changes with mesh size (relative to CPU Baseline):",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        if self.multi_server:
            lines.append(f"> Speedups computed from mean times across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Build header with all solvers except cpu (which is baseline)
        non_baseline_solvers = [s for s in self.solvers if s != "cpu"]
        
        header = "| Model | Size | Nodes |"
        separator = "|-------|------|-------|"
        for solver in non_baseline_solvers:
            header += f" {SOLVER_NAMES.get(solver, solver)} |"
            separator += "----------|"
        
        lines.append(header)
        lines.append(separator)
        
        # Iterate through model+size combinations
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            baseline_mean = solver_stats.get("cpu", {}).get("total_time", {}).get("mean")
            size_label = get_model_size_label(nodes)
            
            row = f"| {model_name} | {size_label} | {format_number(nodes)} |"
            for solver in non_baseline_solvers:
                solver_mean = solver_stats.get(solver, {}).get("total_time", {}).get("mean")
                speedup = calculate_speedup(baseline_mean, solver_mean)
                row += f" {format_speedup(speedup)} |"
            
            lines.append(row)
        
        if len(self.model_size_keys) >= 2:
            lines.extend([
                "",
                "**Observation:** GPU advantage typically increases with problem size due to better utilization of parallel compute units.",
            ])
        
        return "\n".join(lines)
    
    def _generate_convergence(self) -> str:
        """Generate Convergence Verification section."""
        lines = [
            "## Convergence Verification",
            "",
            "All implementations must produce numerically consistent results.",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        # Track if any non-deterministic results found
        non_deterministic_found = False
        
        # Generate convergence info for each model+size
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Solver Convergence",
                "",
                "| Implementation | Converged | Iterations | Final Residual | Relative Residual | N |",
                "|----------------|-----------|------------|----------------|-------------------|---|",
            ])
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats and stats["records"]:
                    # Check convergence (all runs should converge)
                    all_converged = all(r.get("converged", False) for r in stats["records"])
                    converged_str = "Yes" if all_converged else "No"
                    
                    # Iterations - check determinism
                    iter_stats = stats["iterations"]
                    if iter_stats["is_deterministic"]:
                        iter_str = format_number(int(iter_stats["value"])) if iter_stats["value"] else "-"
                    else:
                        non_deterministic_found = True
                        iter_str = f"{int(iter_stats['value']):,} ⚠️"
                    
                    # Residuals - check determinism
                    final_stats = stats["residuals"]["final"]
                    rel_stats = stats["residuals"]["relative"]
                    
                    if final_stats["value"] is not None:
                        final_str = f"{final_stats['value']:.3e}"
                        if not final_stats["is_deterministic"]:
                            final_str += " ⚠️"
                            non_deterministic_found = True
                    else:
                        final_str = "-"
                    
                    if rel_stats["value"] is not None:
                        rel_str = f"{rel_stats['value']:.3e}"
                        if not rel_stats["is_deterministic"]:
                            rel_str += " ⚠️"
                            non_deterministic_found = True
                    else:
                        rel_str = "-"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    n = stats["n"]
                    lines.append(f"| {name} | {converged_str} | {iter_str} | {final_str} | {rel_str} | {n} |")
            
            # Solution statistics (use first record as reference, values should be deterministic)
            lines.extend([
                "",
                "#### Solution Statistics",
                "",
                "| Implementation | u_min | u_max | u_mean |",
                "|----------------|-------|-------|--------|",
            ])
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats and stats["records"]:
                    # Use first record (values should be deterministic)
                    rec = stats["records"][0]
                    sol_stats = rec.get("solution_stats", {})
                    u_range = sol_stats.get("u_range", [None, None])
                    u_mean = sol_stats.get("u_mean")
                    
                    u_min = f"{u_range[0]:.6e}" if u_range[0] is not None else "-"
                    u_max = f"{u_range[1]:.6e}" if u_range[1] is not None else "-"
                    u_mean_str = f"{u_mean:.6e}" if u_mean is not None else "-"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {u_min} | {u_max} | {u_mean_str} |")
            
            lines.append("")
        
        # Add methodology note
        lines.extend([
            "---",
            "",
            "**Methodology Note:** Solver iterations and residuals are verified to be deterministic across all runs. "
            "Results with ⚠️ indicate variance detected and should be reviewed.",
            "",
        ])
        
        if non_deterministic_found:
            lines.append("> ⚠️ **Warning:** Non-deterministic results detected in some runs. This may indicate numerical instability or hardware differences.")
        else:
            lines.append("> ✓ All results verified as deterministic across all runs.")
        
        return "\n".join(lines)
    
    def _generate_efficiency_metrics(self) -> str:
        """Generate Efficiency Metrics section."""
        lines = [
            "## Efficiency Metrics",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        if self.multi_server:
            lines.append(f"> Throughput computed from mean times; memory averaged across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Generate metrics for each model+size
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            size_label = get_model_size_label(nodes)
            
            # Get mesh info from first record
            first_rec = self.records_by_model_size.get((model_name, nodes), [{}])[0]
            elements = first_rec.get("model_elements", 0)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Throughput (computed from mean times)",
                "",
                "| Implementation | Elements/sec | DOFs/sec |",
                "|----------------|--------------|----------|",
            ])
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats:
                    # Get mean assembly and solve times
                    assemble_mean = stats["stages"]["assemble_system"]["mean"] or 0
                    solve_mean = stats["stages"]["solve_system"]["mean"] or 0
                    
                    # Get mean iterations (should be deterministic)
                    iter_val = stats["iterations"]["value"] or 0
                    
                    # Compute throughput from mean times
                    els_per_sec = elements / assemble_mean if assemble_mean > 0 else 0
                    dofs_per_sec = (nodes * iter_val) / solve_mean if solve_mean > 0 else 0
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {els_per_sec:,.0f} | {dofs_per_sec:,.0f} |")
            
            # Memory usage (averaged across runs)
            lines.extend([
                "",
                "#### Memory Usage (mean across runs)",
                "",
                "| Implementation | Peak RAM | Peak VRAM |",
                "|----------------|----------|-----------|",
            ])
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats:
                    ram_mean = stats["memory"]["peak_ram"]["mean"]
                    vram_mean = stats["memory"]["peak_vram"]["mean"]
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    ram_str = format_memory(ram_mean)
                    vram_str = format_memory(vram_mean) if vram_mean else "N/A"
                    lines.append(f"| {name} | {ram_str} | {vram_str} |")
            
            lines.append("")
        
        return "\n".join(lines)
        
        return "\n".join(lines)
    
    def _generate_analysis(self) -> str:
        """Generate Analysis & Discussion section."""
        lines = [
            "## Critical Analysis",
            "",
            "### Bottleneck Evolution",
            "",
            "As optimizations progress, the computational bottleneck shifts:",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        if self.multi_server:
            lines.append(f"> Percentages averaged across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Generate analysis for each model+size
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"#### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "| Implementation | Primary Bottleneck | Secondary Bottleneck |",
                "|----------------|--------------------|---------------------|",
            ])
            
            # Store data for visualization
            solver_stage_pcts = {}
            
            for solver in self.solvers:
                stats = solver_stats.get(solver)
                if stats:
                    total_mean = stats["total_time"]["mean"] or 1
                    
                    # Calculate average percentages for key stages
                    stage_pcts = {
                        "Assembly": (stats["stages"]["assemble_system"]["mean"] or 0) / total_mean * 100,
                        "Solve": (stats["stages"]["solve_system"]["mean"] or 0) / total_mean * 100,
                        "BC": (stats["stages"]["apply_bc"]["mean"] or 0) / total_mean * 100,
                        "Post-Proc": (stats["stages"]["compute_derived"]["mean"] or 0) / total_mean * 100,
                    }
                    
                    solver_stage_pcts[solver] = stage_pcts
                    
                    # Sort by percentage
                    sorted_stages = sorted(stage_pcts.items(), key=lambda x: x[1], reverse=True)
                    
                    primary = f"{sorted_stages[0][0]} ({sorted_stages[0][1]:.0f}%)"
                    secondary = f"{sorted_stages[1][0]} ({sorted_stages[1][1]:.0f}%)"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {primary} | {secondary} |")
            
            # Add text-based visualization (stacked bar)
            lines.extend([
                "",
                "**Time Distribution Visualization:**",
                "",
                "```",
            ])
            
            for solver in self.solvers:
                if solver in solver_stage_pcts:
                    pcts = solver_stage_pcts[solver]
                    name = SOLVER_NAMES.get(solver, solver)
                    
                    # Create a 50-char wide bar
                    bar_width = 50
                    assembly_chars = int(pcts["Assembly"] / 100 * bar_width)
                    solve_chars = int(pcts["Solve"] / 100 * bar_width)
                    bc_chars = int(pcts["BC"] / 100 * bar_width)
                    post_chars = bar_width - assembly_chars - solve_chars - bc_chars
                    
                    bar = "█" * assembly_chars + "▓" * solve_chars + "░" * bc_chars + "·" * max(0, post_chars)
                    
                    lines.append(f"{name:20s} |{bar}|")
            
            lines.extend([
                "```",
                "Legend: █ Assembly  ▓ Solve  ░ BC  · Post-Process",
                "",
            ])
        
        lines.extend([
            "### Why Each Optimization Helps",
            "",
            "| Transition | Reason |",
            "|------------|--------|",
            "| Baseline → Threaded | Limited by Python GIL; threads only help for I/O |",
            "| Threaded → Multiprocess | Bypasses GIL via separate processes; IPC overhead limits gains |",
            "| Multiprocess → Numba CPU | JIT compilation eliminates interpreter overhead; true parallel loops |",
            "| Numba CPU → Numba CUDA | GPU parallelism: thousands of threads vs dozens of CPU cores |",
            "| Numba CUDA → CuPy GPU | CUDA C kernels more optimized than Numba-generated PTX |",
        ])
        
        return "\n".join(lines)
    
    def _generate_conclusions(self) -> str:
        """Generate Conclusions section."""
        lines = [
            "## Conclusions",
            "",
            "### Key Findings",
            "",
        ]
        
        if not self.records:
            lines.append("*No benchmark data available.*")
            return "\n".join(lines)
        
        if self.multi_server:
            lines.append(f"> Conclusions based on mean times across {len(self.server_hashes)} servers.")
            lines.append("")
        
        # Generate findings for each model+size
        for model_name, nodes in self.model_size_keys:
            solver_stats = self._get_solver_stats_for_size(model_name, nodes)
            if not solver_stats:
                continue
            
            size_label = get_model_size_label(nodes)
            
            # Get mean times for each solver
            baseline_mean = solver_stats.get("cpu", {}).get("total_time", {}).get("mean")
            gpu_mean = solver_stats.get("gpu", {}).get("total_time", {}).get("mean")
            numba_mean = solver_stats.get("numba", {}).get("total_time", {}).get("mean")
            threaded_mean = solver_stats.get("cpu_threaded", {}).get("total_time", {}).get("mean")
            
            # Calculate speedups from averaged times
            max_speedup = calculate_speedup(baseline_mean, gpu_mean)
            numba_speedup = calculate_speedup(baseline_mean, numba_mean)
            threaded_speedup = calculate_speedup(baseline_mean, threaded_mean)
            
            lines.append(f"#### {model_name} ({size_label}) - {format_number(nodes)} nodes")
            lines.append("")
            
            finding_num = 1
            
            if max_speedup:
                lines.append(f"{finding_num}. **Maximum Speedup:** CuPy GPU achieves {format_speedup(max_speedup)} speedup over CPU Baseline.")
                lines.append("")
                finding_num += 1
            
            if threaded_speedup:
                lines.append(f"{finding_num}. **Threading Effect:** CPU Threaded shows {format_speedup(threaded_speedup)} speedup.")
                lines.append("")
                finding_num += 1
            
            if numba_speedup:
                lines.append(f"{finding_num}. **JIT Compilation:** Numba CPU delivers {format_speedup(numba_speedup)} speedup by eliminating interpreter overhead.")
                lines.append("")
                finding_num += 1
            
            # GPU solve percentage (from averaged stage times)
            gpu_stats = solver_stats.get("gpu")
            if gpu_stats:
                total_mean = gpu_stats["total_time"]["mean"] or 1
                solve_mean = gpu_stats["stages"]["solve_system"]["mean"] or 0
                solve_pct = solve_mean / total_mean * 100 if total_mean > 0 else 0
                lines.append(f"{finding_num}. **GPU Bottleneck:** On GPU, the iterative solver consumes {solve_pct:.0f}% of total time.")
                lines.append("")
        
        lines.extend([
            "### Recommendations",
            "",
            "| Use Case | Recommended Implementation |",
            "|----------|---------------------------|",
            "| Development/debugging | CPU Baseline or Numba CPU |",
            "| Production (no GPU) | Numba CPU |",
            "| Production (with GPU) | CuPy GPU |",
            "| Small meshes (<10K nodes) | Numba CPU (GPU overhead not worthwhile) |",
            "| Large meshes (>100K nodes) | CuPy GPU |",
        ])
        
        return "\n".join(lines)
    
    def _generate_reproducibility(self) -> str:
        """Generate Reproducibility section."""
        lines = [
            "## Reproducibility",
            "",
            "### Benchmark Metadata",
            "",
            "| Field | Value |",
            "|-------|-------|",
        ]
        
        # Find date range
        timestamps: list[str] = []
        for rec in self.records:
            ts = rec.get("timestamp")
            if ts and isinstance(ts, str):
                timestamps.append(ts)
        
        if timestamps:
            timestamps.sort()
            earliest = timestamps[0][:10]
            latest = timestamps[-1][:10]
            date_range = f"{earliest} to {latest}" if earliest != latest else earliest
        else:
            date_range = "-"
        
        # Collect server info
        servers = collect_unique_servers(self.records)
        
        lines.append(f"| Benchmark Date Range | {date_range} |")
        lines.append(f"| Report Generated | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC |")
        lines.append(f"| Contributing Servers | {len(servers)} |")
        lines.append(f"| Total Records | {len(self.records)} |")
        lines.append(f"| Model/Size Combinations | {len(self.model_size_keys)} |")
        lines.append(f"| Solvers Tested | {len(self.solvers)} |")
        
        # Server details
        if servers:
            lines.extend([
                "",
                "### Contributing Servers",
                "",
                "| Hostname | CPU | GPU | Records |",
                "|----------|-----|-----|---------|",
            ])
            
            for srv in sorted(servers, key=lambda x: x["hostname"]):
                hostname = srv["hostname"].upper()
                cpu = srv["cpu_model"]
                if len(cpu) > 25:
                    cpu = cpu[:22] + "..."
                gpu = srv["gpu_model"] or "N/A"
                if len(gpu) > 20:
                    gpu = gpu[:17] + "..."
                records = srv["record_count"]
                lines.append(f"| {hostname} | {cpu} | {gpu} | {records} |")
        
        # Models included
        lines.extend([
            "",
            "### Models Included",
            "",
            "| Model | Size | Nodes | Records |",
            "|-------|------|-------|---------|",
        ])
        
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            size_label = get_model_size_label(nodes)
            lines.append(f"| {model_name} | {size_label} | {format_number(nodes)} | {len(size_records)} |")
        
        lines.extend([
            "",
            "---",
            "",
            "*FEMulator Pro - GPU-Accelerated Finite Element Analysis*",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_report_generator(
    benchmark_service,
    gallery_file: Optional[Path] = None
) -> ReportGenerator:
    """
    Create a ReportGenerator from a BenchmarkService.
    
    Args:
        benchmark_service: BenchmarkService instance
        gallery_file: Optional path to gallery_files.json
        
    Returns:
        ReportGenerator instance
    """
    # Get records as dicts
    records = [r.to_dict() for r in benchmark_service._records.values()]
    
    # Load gallery data
    gallery_data = {}
    if gallery_file and Path(gallery_file).exists():
        try:
            with open(gallery_file, 'r') as f:
                gallery_data = json.load(f)
        except Exception as e:
            print(f"[Report] Warning: Could not load gallery: {e}")
    
    return ReportGenerator(
        records=records,
        server_config=benchmark_service.server_config,
        gallery_data=gallery_data
    )

def create_report_generator_from_records(
    records: List[Dict[str, Any]],
    server_config: Dict[str, Any],
    gallery_file: Optional[Path] = None
) -> ReportGenerator:
    """
    Create a ReportGenerator from a list of record dicts.
    
    Args:
        records: List of benchmark record dicts (already filtered)
        server_config: Server hardware configuration
        gallery_file: Optional path to gallery_files.json
        
    Returns:
        ReportGenerator instance
    """
    # Load gallery data
    gallery_data = {}
    if gallery_file and Path(gallery_file).exists():
        try:
            with open(gallery_file, 'r') as f:
                gallery_data = json.load(f)
        except Exception as e:
            print(f"[Report] Warning: Could not load gallery: {e}")
    
    return ReportGenerator(
        records=records,
        server_config=server_config,
        gallery_data=gallery_data
    )
