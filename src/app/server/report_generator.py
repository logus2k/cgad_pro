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
    
    def _generate_executive_summary(self) -> str:
        """Generate Executive Summary (Mesh Performance) section."""
        lines = ["Key results from performance benchmarks comparing FEM solver implementations."]
        
        if not self.records:
            lines.append("\n*No benchmark data available.*")
            return "\n".join(lines)
        
        # Show results for each model+size combination
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for THIS specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            baseline_time = best_records.get("cpu", {}).get("timings", {}).get("total_program_time")
            size_label = get_model_size_label(nodes)
            
            lines.append("")
            lines.append(f"**{model_name} ({size_label})** ({format_number(nodes)} nodes)")
            lines.append("")
            lines.append("| Implementation | Total Time | Speedup vs Baseline |")
            lines.append("|----------------|------------|---------------------|")
            
            for solver in self.solvers:
                record = best_records.get(solver)
                if record:
                    time = record.get("timings", {}).get("total_program_time")
                    speedup = calculate_speedup(baseline_time, time) if baseline_time else None
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    time_str = format_time(time)
                    speedup_str = format_speedup(speedup) if solver != "cpu" else "1.0x"
                    
                    lines.append(f"| {name} | {time_str} | {speedup_str} |")
        
        return "\n".join(lines)
    
    def _generate_test_configuration(self) -> str:
        """Generate Test Configuration section."""
        lines = [
            "## Hardware Environment",
            "",
            "| Component | Specification |",
            "|-----------|---------------|",
        ]
        
        # Hardware table
        config = self.server_config
        lines.append(f"| CPU | {config.get('cpu_model', 'Unknown')} ({config.get('cpu_cores', '?')} cores) |")
        lines.append(f"| RAM | {config.get('ram_gb', '?')} GB |")
        
        if config.get('gpu_model'):
            lines.append(f"| GPU | {config.get('gpu_model')} ({config.get('gpu_memory_gb', '?')} GB VRAM) |")
        
        if config.get('cuda_version'):
            lines.append(f"| CUDA Version | {config.get('cuda_version')} |")
        
        lines.append(f"| OS | {config.get('os', 'Unknown')} |")
        lines.append(f"| Python | {config.get('python_version', 'Unknown')} |")
        
        # Test Meshes
        lines.extend([
            "",
            "### Test Meshes",
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
            "### Solver Configuration",
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
            "### Implementations Tested",
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
        
        # Summary table per model+size
        lines.extend([
            "### Summary Table",
            "",
            "Total workflow time (best run per solver):",
            "",
        ])
        
        # Build header with model+size combinations
        header = "| Implementation |"
        separator = "|----------------|"
        for model_name, nodes in self.model_size_keys:
            size_label = get_model_size_label(nodes)
            # Shorten model name for header
            short_model = model_name[:8] if len(model_name) > 8 else model_name
            header += f" {short_model} {size_label} |"
            separator += "------------|"
        
        lines.append(header)
        lines.append(separator)
        
        # Build rows - one per solver
        for solver in self.solvers:
            row = f"| {SOLVER_NAMES.get(solver, solver)} |"
            for model_name, nodes in self.model_size_keys:
                size_records = self.records_by_model_size.get((model_name, nodes), [])
                # Get best time for this solver in this model+size
                best_time = None
                for rec in size_records:
                    if rec.get("solver_type") == solver:
                        time = rec.get("timings", {}).get("total_program_time")
                        if time is not None and (best_time is None or time < best_time):
                            best_time = time
                row += f" {format_time(best_time)} |"
            lines.append(row)
        
        # Speedup table
        lines.extend([
            "",
            "### Speedup vs CPU Baseline",
            "",
        ])
        
        header = "| Implementation |"
        separator = "|----------------|"
        for model_name, nodes in self.model_size_keys:
            size_label = get_model_size_label(nodes)
            header += f" {size_label} |"
            separator += "-----|"
        
        lines.append(header)
        lines.append(separator)
        
        for solver in self.solvers:
            row = f"| {SOLVER_NAMES.get(solver, solver)} |"
            for model_name, nodes in self.model_size_keys:
                size_records = self.records_by_model_size.get((model_name, nodes), [])
                # Get best times for baseline and this solver
                baseline_time = None
                solver_time = None
                for rec in size_records:
                    time = rec.get("timings", {}).get("total_program_time")
                    if time is None:
                        continue
                    if rec.get("solver_type") == "cpu":
                        if baseline_time is None or time < baseline_time:
                            baseline_time = time
                    if rec.get("solver_type") == solver:
                        if solver_time is None or time < solver_time:
                            solver_time = time
                
                if baseline_time and solver_time:
                    speedup = calculate_speedup(baseline_time, solver_time)
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
        
        # Generate breakdown for each model+size combination
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for this specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Stage Timings (seconds)",
                "",
            ])
            
            # Stage timing table
            header = "| Stage |"
            separator = "|-------|"
            for solver in self.solvers:
                if solver in best_records:
                    header += f" {SOLVER_NAMES.get(solver, solver)} |"
                    separator += "----------|"
            
            lines.append(header)
            lines.append(separator)
            
            for stage in stages:
                label = stage_labels.get(stage, stage)
                row = f"| {label} |"
                for solver in self.solvers:
                    rec = best_records.get(solver)
                    if rec:
                        time = rec.get("timings", {}).get(stage)
                        row += f" {format_time(time)} |"
                lines.append(row)
            
            # Time distribution
            lines.extend([
                "",
                "#### Time Distribution (% of Total)",
                "",
            ])
            
            header = "| Implementation |"
            separator = "|----------------|"
            for stage in stages[:-1]:  # Exclude total
                header += f" {stage_labels.get(stage, stage)} |"
                separator += "------|"
            
            lines.append(header)
            lines.append(separator)
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if not rec:
                    continue
                row = f"| {SOLVER_NAMES.get(solver, solver)} |"
                total = rec.get("timings", {}).get("total_program_time", 1)
                for stage in stages[:-1]:
                    time = rec.get("timings", {}).get(stage, 0)
                    pct = (time / total * 100) if total > 0 else 0
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
        
        # Build header with all solvers except cpu (which is baseline)
        non_baseline_solvers = [s for s in self.solvers if s != "cpu"]
        
        header = "| Model | Size | Nodes |"
        separator = "|-------|------|-------|"
        for solver in non_baseline_solvers:
            header += f" {SOLVER_NAMES.get(solver, solver)} |"
            separator += "----------|"
        
        lines.append(header)
        lines.append(separator)
        
        # Iterate through model+size combinations (already sorted by model then nodes)
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best time per solver for this model+size
            best_times = {}
            for rec in size_records:
                solver = rec.get("solver_type")
                time = rec.get("timings", {}).get("total_program_time")
                if solver and time is not None:
                    if solver not in best_times or time < best_times[solver]:
                        best_times[solver] = time
            
            baseline = best_times.get("cpu")
            size_label = get_model_size_label(nodes)
            
            row = f"| {model_name} | {size_label} | {format_number(nodes)} |"
            for solver in non_baseline_solvers:
                solver_time = best_times.get(solver)
                speedup = calculate_speedup(baseline, solver_time)
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
        
        # Generate convergence info for each model+size
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for this specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Solver Convergence",
                "",
                "| Implementation | Converged | Iterations | Final Residual | Relative Residual |",
                "|----------------|-----------|------------|----------------|-------------------|",
            ])
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if rec:
                    converged = "Yes" if rec.get("converged") else "No"
                    iterations = format_number(rec.get("iterations"))
                    
                    stats = rec.get("solution_stats", {})
                    final_res = stats.get("final_residual")
                    rel_res = stats.get("relative_residual")
                    
                    final_str = f"{final_res:.3e}" if final_res is not None else "-"
                    rel_str = f"{rel_res:.3e}" if rel_res is not None else "-"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {converged} | {iterations} | {final_str} | {rel_str} |")
            
            # Solution statistics
            lines.extend([
                "",
                "#### Solution Statistics",
                "",
                "| Implementation | u_min | u_max | u_mean |",
                "|----------------|-------|-------|--------|",
            ])
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if rec:
                    stats = rec.get("solution_stats", {})
                    u_range = stats.get("u_range", [None, None])
                    u_mean = stats.get("u_mean")
                    
                    u_min = f"{u_range[0]:.6e}" if u_range[0] is not None else "-"
                    u_max = f"{u_range[1]:.6e}" if u_range[1] is not None else "-"
                    u_mean_str = f"{u_mean:.6e}" if u_mean is not None else "-"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {u_min} | {u_max} | {u_mean_str} |")
            
            lines.append("")
        
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
        
        # Generate metrics for each model+size
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for this specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "#### Throughput",
                "",
                "| Implementation | Elements/sec | DOFs/sec |",
                "|----------------|--------------|----------|",
            ])
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if rec:
                    elements = rec.get("model_elements", 0)
                    rec_nodes = rec.get("model_nodes", 0)
                    iterations = rec.get("iterations", 0)
                    
                    assemble_time = rec.get("timings", {}).get("assemble_system", 0)
                    solve_time = rec.get("timings", {}).get("solve_system", 0)
                    
                    els_per_sec = elements / assemble_time if assemble_time > 0 else 0
                    dofs_per_sec = (rec_nodes * iterations) / solve_time if solve_time > 0 else 0
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {els_per_sec:,.0f} | {dofs_per_sec:,.0f} |")
            
            # Memory usage
            lines.extend([
                "",
                "#### Memory Usage",
                "",
                "| Implementation | Peak RAM | Peak VRAM |",
                "|----------------|----------|-----------|",
            ])
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if rec:
                    memory = rec.get("memory", {})
                    peak_ram = memory.get("peak_ram_mb")
                    peak_vram = memory.get("peak_vram_mb")
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    ram_str = format_memory(peak_ram)
                    vram_str = format_memory(peak_vram) if peak_vram else "N/A"
                    lines.append(f"| {name} | {ram_str} | {vram_str} |")
            
            lines.append("")
        
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
        
        # Generate analysis for each model+size
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for this specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            size_label = get_model_size_label(nodes)
            
            lines.extend([
                f"#### {model_name} ({size_label}) - {format_number(nodes)} nodes",
                "",
                "| Implementation | Primary Bottleneck | Secondary Bottleneck |",
                "|----------------|--------------------|---------------------|",
            ])
            
            for solver in self.solvers:
                rec = best_records.get(solver)
                if rec:
                    timings = rec.get("timings", {})
                    total = timings.get("total_program_time", 1)
                    
                    # Calculate percentages for key stages
                    stages = {
                        "Assembly": timings.get("assemble_system", 0) / total * 100 if total > 0 else 0,
                        "Solve": timings.get("solve_system", 0) / total * 100 if total > 0 else 0,
                        "BC": timings.get("apply_bc", 0) / total * 100 if total > 0 else 0,
                        "Post-Proc": timings.get("compute_derived", 0) / total * 100 if total > 0 else 0,
                    }
                    
                    # Sort by percentage
                    sorted_stages = sorted(stages.items(), key=lambda x: x[1], reverse=True)
                    
                    primary = f"{sorted_stages[0][0]} ({sorted_stages[0][1]:.0f}%)"
                    secondary = f"{sorted_stages[1][0]} ({sorted_stages[1][1]:.0f}%)"
                    
                    name = SOLVER_NAMES.get(solver, solver)
                    lines.append(f"| {name} | {primary} | {secondary} |")
            
            lines.append("")
        
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
        
        # Generate findings for each model+size
        for model_name, nodes in self.model_size_keys:
            size_records = self.records_by_model_size.get((model_name, nodes), [])
            if not size_records:
                continue
            
            # Get best record per solver for this specific mesh size
            best_records = {}
            for record in size_records:
                solver = record.get("solver_type")
                if not solver:
                    continue
                total_time = record.get("timings", {}).get("total_program_time")
                if total_time is None:
                    continue
                if solver not in best_records or total_time < best_records[solver].get("timings", {}).get("total_program_time", float("inf")):
                    best_records[solver] = record
            
            if not best_records:
                continue
            
            size_label = get_model_size_label(nodes)
            
            baseline_time = best_records.get("cpu", {}).get("timings", {}).get("total_program_time")
            gpu_time = best_records.get("gpu", {}).get("timings", {}).get("total_program_time")
            numba_time = best_records.get("numba", {}).get("timings", {}).get("total_program_time")
            threaded_time = best_records.get("cpu_threaded", {}).get("timings", {}).get("total_program_time")
            
            max_speedup = calculate_speedup(baseline_time, gpu_time)
            numba_speedup = calculate_speedup(baseline_time, numba_time)
            threaded_speedup = calculate_speedup(baseline_time, threaded_time)
            
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
            
            # GPU solve percentage
            gpu_rec = best_records.get("gpu")
            if gpu_rec:
                timings = gpu_rec.get("timings", {})
                total = timings.get("total_program_time", 1)
                solve_pct = timings.get("solve_system", 0) / total * 100 if total > 0 else 0
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
        
        lines.append(f"| Benchmark Date Range | {date_range} |")
        lines.append(f"| Report Generated | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC |")
        lines.append(f"| Server Hostname | {self.server_config.get('hostname', 'Unknown')} |")
        lines.append(f"| Total Records | {len(self.records)} |")
        lines.append(f"| Model/Size Combinations | {len(self.model_size_keys)} |")
        lines.append(f"| Solvers Tested | {len(self.solvers)} |")
        
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
