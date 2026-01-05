#!/usr/bin/env python3
"""
FEMulator Pro - Automated Benchmark Runner

CLI tool for running automated benchmarks across all solver implementations.

Usage:
    python run_benchmark.py
    python run_benchmark.py --solver gpu
    python run_benchmark.py --model Venturi
    python run_benchmark.py --max-nodes 1000
    python run_benchmark.py --dry-run

Location: /src/app/server/automated_benchmark/run_benchmark.py
"""

import sys
import argparse
from pathlib import Path

# Setup paths
HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent                    # /src/app/server
SRC_DIR = SERVER_DIR.parent.parent          # /src

# Add solver directories to path
for package in ["cpu", "cpu_threaded", "cpu_multiprocess", "numba", "numba_cuda", "gpu", "shared"]:
    package_path = SRC_DIR / package
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

# Add server directory for solver_wrapper and memory_tracker imports
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

# Add this package directory
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config_loader import ConfigLoader
from result_recorder import ResultRecorder
from runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="FEMulator Pro Automated Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                        # Full benchmark suite
  python run_benchmark.py --solver gpu           # GPU solver only
  python run_benchmark.py --model Venturi        # Venturi model only  
  python run_benchmark.py --max-nodes 1000       # Small meshes only
  python run_benchmark.py --dry-run              # Preview without running
  python run_benchmark.py --runs 5               # 5 runs per test
  python run_benchmark.py --clear                # Clear previous results first

Available solvers:
  cpu, cpu_threaded, cpu_multiprocess, numba, numba_cuda, gpu

Available models:
  Y-Shaped, Venturi, S-Bend, T-Junction, Backward-Facing Step, Elbow 90
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to testing_procedure.json (default: ./testing_procedure.json)"
    )
    
    parser.add_argument(
        "--gallery", "-g",
        type=str,
        default=None,
        help="Override gallery file path"
    )
    
    parser.add_argument(
        "--solver", "-s",
        type=str,
        default=None,
        choices=["cpu", "cpu_threaded", "cpu_multiprocess", "numba", "numba_cuda", "gpu"],
        help="Run only this solver"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Run only this model (e.g., 'Venturi')"
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum mesh size (nodes) to test"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=None,
        help="Override number of runs per test"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show test matrix without executing"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing automated benchmark records before running"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override benchmark output directory"
    )
    
    args = parser.parse_args()
    
    # Resolve config path (default: same directory as this script)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
    else:
        config_path = HERE / "testing_procedure.json"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    gallery_path = Path(args.gallery) if args.gallery else None
    
    # Load configuration
    try:
        config = ConfigLoader(config_path, gallery_override=gallery_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Override runs if specified
    if args.runs:
        config.execution.runs_per_test = args.runs
    
    # Output directory: /src/app/server/benchmark/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SERVER_DIR / "benchmark"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result recorder
    recorder = ResultRecorder(output_dir)
    
    # Clear existing records if requested
    if args.clear:
        count = recorder.clear_records()
        print(f"Cleared {count} existing automated benchmark records")
    
    # Mesh base directory: /src/app/client/
    mesh_base_dir = SRC_DIR / "app" / "client"
    
    if not mesh_base_dir.exists():
        print(f"Error: Client directory not found: {mesh_base_dir}")
        sys.exit(1)
    
    # Initialize runner
    runner = BenchmarkRunner(config, recorder, mesh_base_dir)
    
    # Execute benchmark
    results, error = runner.run(
        solver_filter=args.solver,
        model_filter=args.model,
        max_nodes=args.max_nodes,
        dry_run=args.dry_run
    )
    
    if error and not args.dry_run:
        print(f"\nBenchmark failed: {error}")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
