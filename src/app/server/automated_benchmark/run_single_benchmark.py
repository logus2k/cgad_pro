#!/usr/bin/env python3
"""
FEMulator Pro - Single Benchmark Runner

Minimal script for running a single solver/mesh combination.
Designed to be wrapped by nsys/ncu for profiling.

Usage:
    python run_single_benchmark.py --solver gpu --mesh path/to/mesh.h5
    nsys profile -o output python run_single_benchmark.py --solver gpu --mesh mesh.h5

Location: /src/app/server/automated_benchmark/run_single_benchmark.py
"""

import sys
import json
import argparse
import time
from pathlib import Path

# Setup paths (same as run_benchmark.py)
HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
SRC_DIR = SERVER_DIR.parent.parent

# Add solver directories to path
for package in ["cpu", "cpu_threaded", "cpu_multiprocess", "numba", "numba_cuda", "gpu", "shared"]:
    package_path = SRC_DIR / package
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

# Add server directory
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


# Solver class mapping
SOLVER_CLASSES = {
    "cpu": ("quad8_cpu_v4", "Quad8FEMSolver"),
    "cpu_threaded": ("quad8_cpu_threaded_v2", "Quad8FEMSolverThreaded"),
    "cpu_multiprocess": ("quad8_cpu_multiprocess_v3", "Quad8FEMSolverMultiprocess"),
    "numba": ("quad8_numba_v2", "Quad8FEMSolverNumba"),
    "numba_cuda": ("quad8_numba_cuda", "Quad8FEMSolverNumbaCUDA"),
    "gpu": ("quad8_gpu_v3", "Quad8FEMSolverGPU"),
}


def get_solver_class(solver_type: str):
    """Dynamically import and return solver class."""
    if solver_type not in SOLVER_CLASSES:
        raise ValueError(f"Unknown solver: {solver_type}. Valid: {list(SOLVER_CLASSES.keys())}")
    
    module_name, class_name = SOLVER_CLASSES[solver_type]
    module = __import__(module_name)
    return getattr(module, class_name)


def resolve_mesh_path(mesh_file: str) -> Path:
    """Resolve mesh file path."""
    mesh_path = Path(mesh_file)
    
    # If absolute and exists, use it
    if mesh_path.is_absolute() and mesh_path.exists():
        return mesh_path
    
    # Try relative to CWD
    if mesh_path.exists():
        return mesh_path.resolve()
    
    # Try relative to client directory
    client_path = SRC_DIR / "app" / "client" / mesh_file
    if client_path.exists():
        return client_path
    
    # Try relative to client/mesh directory
    mesh_dir_path = SRC_DIR / "app" / "client" / "mesh" / mesh_path.name
    if mesh_dir_path.exists():
        return mesh_dir_path
    
    raise FileNotFoundError(f"Mesh file not found: {mesh_file}")


def run_single(
    solver_type: str,
    mesh_file: str,
    max_iterations: int = 15000,
    tolerance: float = 1e-8,
    verbose: bool = True
) -> dict:
    """
    Run a single solver execution.
    
    Returns dict with timing metrics and solution stats.
    """
    # Resolve mesh path
    mesh_path = resolve_mesh_path(mesh_file)
    
    if verbose:
        print(f"{'='*60}")
        print(f"Single Benchmark Run")
        print(f"{'='*60}")
        print(f"Solver: {solver_type}")
        print(f"Mesh:   {mesh_path}")
        print(f"{'='*60}")
    
    # Get solver class
    SolverClass = get_solver_class(solver_type)
    
    # Run solver
    t_start = time.perf_counter()
    
    solver = SolverClass(
        mesh_file=str(mesh_path),
        maxiter=max_iterations,
        rtol=tolerance,
        verbose=verbose,
        progress_callback=None
    )
    
    results = solver.run()
    
    t_end = time.perf_counter()
    
    # Build output
    output = {
        "solver_type": solver_type,
        "mesh_file": str(mesh_path),
        "mesh_name": mesh_path.stem,
        "converged": results.get("converged", False),
        "iterations": results.get("iterations", 0),
        "timing_metrics": results.get("timing_metrics", {}),
        "solution_stats": results.get("solution_stats", {}),
        "mesh_info": results.get("mesh_info", {}),
        "total_time": t_end - t_start
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Converged:  {output['converged']}")
        print(f"Iterations: {output['iterations']}")
        print(f"Total time: {output['total_time']:.3f}s")
        print(f"\nTiming breakdown:")
        for key, val in output['timing_metrics'].items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}s")
        print(f"{'='*60}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run single FEM solver benchmark (for profiling)"
    )
    
    parser.add_argument(
        "--solver", "-s",
        type=str,
        required=True,
        choices=list(SOLVER_CLASSES.keys()),
        help="Solver type"
    )
    
    parser.add_argument(
        "--mesh", "-m",
        type=str,
        required=True,
        help="Path to mesh HDF5 file"
    )
    
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=15000,
        help="Maximum solver iterations (default: 15000)"
    )
    
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-8,
        help="Solver tolerance (default: 1e-8)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_single(
            solver_type=args.solver,
            mesh_file=args.mesh,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            verbose=not args.quiet
        )
        
        if args.json:
            print("\n--- JSON OUTPUT ---")
            print(json.dumps(results, indent=2, default=str))
        
        sys.exit(0 if results["converged"] else 1)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
