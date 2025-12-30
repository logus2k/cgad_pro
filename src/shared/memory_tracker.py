"""
Memory Tracker for FEM Solver Benchmarking.

Provides continuous background sampling of RAM and VRAM usage during solver
execution, capturing true peak memory consumption without modifying solver code.

Features:
- Daemon thread with guaranteed cleanup (no zombies)
- Configurable sampling interval
- Thread-safe peak value tracking
- Support for both CPU (psutil) and GPU (CuPy) memory
- Context manager and wrapper function APIs

Location: /src/shared/memory_tracker.py

Usage:
    # Option 1: Context manager
    with MemoryTracker() as tracker:
        results = solver.run()
    memory = tracker.get_peak()
    
    # Option 2: Wrapper function
    output = run_with_memory_tracking(solver)
    results = output["results"]
    memory = output["memory"]
"""

import os
import sys
import time
import atexit
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from weakref import WeakSet

import psutil

# =============================================================================
# Configuration
# =============================================================================

# Default sampling interval in milliseconds
DEFAULT_SAMPLING_INTERVAL_MS = 100

# =============================================================================
# GPU Memory Detection
# =============================================================================

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # type: ignore


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB (CuPy memory pool)."""
    if not HAS_CUPY:
        return 0.0
    try:
        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / (1024 * 1024)
    except Exception:
        return 0.0


def get_gpu_total_memory_mb() -> float:
    """Get total GPU memory allocated (including cached) in MB."""
    if not HAS_CUPY:
        return 0.0
    try:
        pool = cp.get_default_memory_pool()
        return pool.total_bytes() / (1024 * 1024)
    except Exception:
        return 0.0


# =============================================================================
# Global Tracker Registry (for emergency cleanup)
# =============================================================================

_active_trackers: WeakSet = WeakSet()


@atexit.register
def _cleanup_all_trackers():
    """Stop all active trackers on process exit."""
    for tracker in list(_active_trackers):
        try:
            tracker.stop()
        except Exception:
            pass


# =============================================================================
# Memory Tracker Class
# =============================================================================

class MemoryTracker:
    """
    Continuous background memory tracker with guaranteed cleanup.
    
    Samples RAM and VRAM at regular intervals in a daemon thread,
    tracking peak values throughout the measurement period.
    
    Thread Safety:
    - All public methods are thread-safe
    - Peak values protected by lock
    - Daemon thread ensures no zombies on process exit
    
    Example:
        tracker = MemoryTracker(interval_ms=100)
        tracker.start()
        # ... run computation ...
        memory = tracker.stop()
        print(f"Peak RAM: {memory['peak_ram_mb']:.1f} MB")
    """
    
    def __init__(self, interval_ms: int = DEFAULT_SAMPLING_INTERVAL_MS):
        """
        Initialize memory tracker.
        
        Args:
            interval_ms: Sampling interval in milliseconds (default: 100)
        """
        self.interval_ms = interval_ms
        self._interval_sec = interval_ms / 1000.0
        
        # Process handle for RAM measurement
        self._process = psutil.Process(os.getpid())
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Baseline values (captured at start)
        self._baseline_ram_mb = 0.0
        self._baseline_vram_mb = 0.0
        
        # Peak values (updated during sampling)
        self._peak_ram_mb = 0.0
        self._peak_vram_mb = 0.0
        
        # Sample count (for diagnostics)
        self._sample_count = 0
        
        # Register for emergency cleanup
        _active_trackers.add(self)
    
    def start(self) -> 'MemoryTracker':
        """
        Start background memory sampling.
        
        Returns:
            self (for chaining)
        """
        with self._lock:
            if self._running:
                return self  # Already running, idempotent
            
            # Capture baseline
            self._baseline_ram_mb = self._get_ram_mb()
            self._baseline_vram_mb = get_gpu_memory_mb()
            
            # Reset peaks
            self._peak_ram_mb = 0.0
            self._peak_vram_mb = 0.0
            self._sample_count = 0
            
            # Start daemon thread
            self._running = True
            self._thread = threading.Thread(
                target=self._sample_loop,
                daemon=True,  # Critical: killed if main process exits
                name="MemoryTracker"
            )
            self._thread.start()
        
        return self
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop sampling and return peak memory values.
        
        Returns:
            dict with peak_ram_mb, peak_vram_mb, and metadata
        """
        with self._lock:
            self._running = False
        
        # Wait for thread to finish (with timeout to prevent hangs)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
            if self._thread.is_alive():
                # Thread didn't stop cleanly, but it's a daemon so it will
                # be killed when the process exits
                print("[MemoryTracker] Warning: sampling thread did not stop cleanly")
        
        self._thread = None
        
        # Return results
        with self._lock:
            return {
                "peak_ram_mb": round(self._peak_ram_mb, 2),
                "peak_vram_mb": round(self._peak_vram_mb, 2),
                "baseline_ram_mb": round(self._baseline_ram_mb, 2),
                "baseline_vram_mb": round(self._baseline_vram_mb, 2),
                "sample_count": self._sample_count,
                "interval_ms": self.interval_ms,
                "gpu_available": HAS_CUPY
            }
    
    def get_peak(self) -> Dict[str, float]:
        """
        Get current peak values without stopping.
        
        Returns:
            dict with peak_ram_mb and peak_vram_mb
        """
        with self._lock:
            return {
                "peak_ram_mb": round(self._peak_ram_mb, 2),
                "peak_vram_mb": round(self._peak_vram_mb, 2)
            }
    
    def _get_ram_mb(self) -> float:
        """Get current process RAM usage in MB."""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _sample_loop(self):
        """Background sampling loop (runs in daemon thread)."""
        while self._running:
            try:
                # Sample RAM (relative to baseline)
                current_ram = self._get_ram_mb() - self._baseline_ram_mb
                
                # Sample VRAM (relative to baseline)
                current_vram = get_gpu_memory_mb() - self._baseline_vram_mb
                
                # Update peaks (thread-safe)
                with self._lock:
                    if current_ram > self._peak_ram_mb:
                        self._peak_ram_mb = current_ram
                    if current_vram > self._peak_vram_mb:
                        self._peak_vram_mb = current_vram
                    self._sample_count += 1
                
            except Exception:
                # Silently ignore sampling errors to avoid disrupting solver
                pass
            
            # Sleep until next sample
            time.sleep(self._interval_sec)
    
    def __enter__(self) -> 'MemoryTracker':
        """Context manager entry - starts tracking."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops tracking."""
        self.stop()
        return False  # Don't suppress exceptions
    
    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        if self._running:
            self.stop()


# =============================================================================
# Wrapper Function API
# =============================================================================

def run_with_memory_tracking(
    solver,
    interval_ms: int = DEFAULT_SAMPLING_INTERVAL_MS,
    **run_kwargs
) -> Dict[str, Any]:
    """
    Run a solver with memory tracking enabled.
    
    This is the primary API for integrating memory tracking with existing
    solver code without any modifications to the solver itself.
    
    Args:
        solver: Any FEM solver instance with a run() method
        interval_ms: Memory sampling interval in milliseconds
        **run_kwargs: Additional arguments to pass to solver.run()
    
    Returns:
        dict with:
            - "results": The solver's return value
            - "memory": Peak memory statistics
    
    Example:
        solver = Quad8FEMSolverGPU(mesh_file="mesh.h5")
        output = run_with_memory_tracking(solver)
        print(f"Solve time: {output['results']['timing_metrics']['total_workflow']:.2f}s")
        print(f"Peak VRAM: {output['memory']['peak_vram_mb']:.1f} MB")
    """
    tracker = MemoryTracker(interval_ms=interval_ms)
    tracker.start()
    
    try:
        results = solver.run(**run_kwargs)
    finally:
        memory = tracker.stop()
    
    return {
        "results": results,
        "memory": memory
    }


# =============================================================================
# Solver Factory
# =============================================================================

# Solver implementation registry
SOLVER_IMPLEMENTATIONS = {
    "cpu": {
        "module": "quad8_cpu_v3",
        "class": "Quad8FEMSolver",
        "package": "cpu",
        "description": "CPU Baseline (Sequential Python)"
    },
    "cpu_threaded": {
        "module": "quad8_cpu_threaded",
        "class": "Quad8FEMSolverThreaded",
        "package": "cpu_threaded",
        "description": "CPU Threaded (ThreadPoolExecutor)"
    },
    "cpu_multiprocess": {
        "module": "quad8_cpu_multiprocess",
        "class": "Quad8FEMSolverMultiprocess",
        "package": "cpu_multiprocess",
        "description": "CPU Multiprocess (multiprocessing.Pool)"
    },
    "numba": {
        "module": "quad8_numba",
        "class": "Quad8FEMSolverNumba",
        "package": "numba",
        "description": "Numba JIT CPU (@njit + prange)"
    },
    "numba_cuda": {
        "module": "quad8_numba_cuda",
        "class": "Quad8FEMSolverNumbaCUDA",
        "package": "numba_cuda",
        "description": "Numba CUDA (@cuda.jit kernels)"
    },
    "gpu": {
        "module": "quad8_gpu_v3",
        "class": "Quad8FEMSolverGPU",
        "package": "gpu",
        "description": "CuPy GPU (CUDA C RawKernels)"
    }
}


def get_solver_class(implementation: str):
    """
    Dynamically import and return a solver class.
    
    Args:
        implementation: One of the keys in SOLVER_IMPLEMENTATIONS
    
    Returns:
        The solver class
    
    Raises:
        ValueError: If implementation is not recognized
        ImportError: If the solver module cannot be imported
    """
    if implementation not in SOLVER_IMPLEMENTATIONS:
        valid = ", ".join(SOLVER_IMPLEMENTATIONS.keys())
        raise ValueError(f"Unknown implementation '{implementation}'. Valid options: {valid}")
    
    config = SOLVER_IMPLEMENTATIONS[implementation]
    
    # Import the module
    module = __import__(config["module"], fromlist=[config["class"]])
    
    # Get the class
    solver_class = getattr(module, config["class"])
    
    return solver_class


def create_solver(
    implementation: str,
    mesh_file: str,
    maxiter: int = 15000,
    verbose: bool = True,
    **kwargs
):
    """
    Create a solver instance for the specified implementation.
    
    Args:
        implementation: One of: cpu, cpu_threaded, cpu_multiprocess, numba, numba_cuda, gpu
        mesh_file: Path to mesh file (.h5)
        maxiter: Maximum solver iterations
        verbose: Enable progress output
        **kwargs: Additional solver parameters
    
    Returns:
        Configured solver instance
    """
    solver_class = get_solver_class(implementation)
    
    return solver_class(
        mesh_file=mesh_file,
        maxiter=maxiter,
        verbose=verbose,
        **kwargs
    )


# =============================================================================
# Main - Test Script
# =============================================================================

def main():
    """
    Test memory tracking with selectable solver implementation.
    
    Usage:
        python memory_tracker.py [implementation] [mesh_file]
    
    Arguments:
        implementation: cpu, cpu_threaded, cpu_multiprocess, numba, numba_cuda, gpu (default: gpu)
        mesh_file: Path to mesh file (default: searches for y_tube1_1_3m.h5)
    
    Examples:
        python memory_tracker.py gpu
        python memory_tracker.py numba /path/to/mesh.h5
        python memory_tracker.py cpu_multiprocess y_tube1_195k.h5
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test memory tracking with FEM solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available implementations:
  cpu              CPU Baseline (Sequential Python)
  cpu_threaded     CPU Threaded (ThreadPoolExecutor)
  cpu_multiprocess CPU Multiprocess (multiprocessing.Pool)
  numba            Numba JIT CPU (@njit + prange)
  numba_cuda       Numba CUDA (@cuda.jit kernels)
  gpu              CuPy GPU (CUDA C RawKernels) [default]
        """
    )
    parser.add_argument(
        "implementation",
        nargs="?",
        default="gpu",
        choices=list(SOLVER_IMPLEMENTATIONS.keys()),
        help="Solver implementation to test (default: gpu)"
    )
    parser.add_argument(
        "mesh_file",
        nargs="?",
        default=None,
        help="Path to mesh file (default: auto-detect y_tube1_1_3m.h5)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_SAMPLING_INTERVAL_MS,
        help=f"Memory sampling interval in ms (default: {DEFAULT_SAMPLING_INTERVAL_MS})"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=15000,
        help="Maximum solver iterations (default: 15000)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce solver output verbosity"
    )
    
    args = parser.parse_args()
    
    # Find mesh file
    mesh_file = args.mesh_file
    if mesh_file is None:
        # Try to auto-detect mesh file location
        possible_paths = [
            Path("y_tube1_1_3m.h5"),
            Path("mesh/y_tube1_1_3m.h5"),
            Path("../client/mesh/y_tube1_1_3m.h5"),
            Path(__file__).parent.parent / "app/client/mesh/y_tube1_1_3m.h5",
            Path("/src/app/client/mesh/y_tube1_1_3m.h5"),
        ]
        
        for p in possible_paths:
            if p.exists():
                mesh_file = str(p)
                break
        
        if mesh_file is None:
            print("ERROR: Could not find default mesh file (y_tube1_1_3m.h5)")
            print("Please specify mesh file path as argument:")
            print(f"  python {sys.argv[0]} {args.implementation} /path/to/mesh.h5")
            sys.exit(1)
    
    mesh_path = Path(mesh_file)
    if not mesh_path.exists():
        print(f"ERROR: Mesh file not found: {mesh_file}")
        sys.exit(1)
    
    # Print configuration
    impl_info = SOLVER_IMPLEMENTATIONS[args.implementation]
    print("=" * 70)
    print("MEMORY TRACKING TEST")
    print("=" * 70)
    print(f"Implementation:    {args.implementation} - {impl_info['description']}")
    print(f"Mesh file:         {mesh_path.name}")
    print(f"Mesh path:         {mesh_path.absolute()}")
    print(f"Sampling interval: {args.interval} ms")
    print(f"Max iterations:    {args.maxiter}")
    print(f"GPU available:     {HAS_CUPY}")
    print("=" * 70)
    print()
    
    # Setup Python path for solver imports
    src_root = Path(__file__).parent.parent
    for package in ["cpu", "cpu_threaded", "cpu_multiprocess", "numba", "numba_cuda", "gpu", "shared"]:
        package_path = src_root / package
        if package_path.exists() and str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))
    
    # Create solver
    print(f"Creating {args.implementation} solver...")
    try:
        solver = create_solver(
            implementation=args.implementation,
            mesh_file=str(mesh_path),
            maxiter=args.maxiter,
            verbose=not args.quiet
        )
    except ImportError as e:
        print(f"ERROR: Could not import solver: {e}")
        print(f"Make sure you're running from the correct directory")
        sys.exit(1)
    
    # Run with memory tracking
    print(f"\nStarting solver with memory tracking (interval={args.interval}ms)...")
    print("-" * 70)
    
    output = run_with_memory_tracking(
        solver,
        interval_ms=args.interval
    )
    
    results = output["results"]
    memory = output["memory"]
    
    # Print results
    print("-" * 70)
    print("\nRESULTS")
    print("=" * 70)
    
    # Solver results
    print("\nSolver Performance:")
    print(f"  Converged:       {results.get('converged', 'N/A')}")
    print(f"  Iterations:      {results.get('iterations', 'N/A')}")
    
    timing = results.get('timing_metrics', {})
    print(f"\nTiming Breakdown:")
    for stage, duration in timing.items():
        print(f"  {stage:<20} {duration:>10.4f}s")
    
    # Memory results
    print(f"\nMemory Usage:")
    print(f"  Peak RAM:        {memory['peak_ram_mb']:>10.2f} MB")
    print(f"  Peak VRAM:       {memory['peak_vram_mb']:>10.2f} MB")
    print(f"  Baseline RAM:    {memory['baseline_ram_mb']:>10.2f} MB")
    print(f"  Baseline VRAM:   {memory['baseline_vram_mb']:>10.2f} MB")
    print(f"  Samples taken:   {memory['sample_count']:>10}")
    
    # Solution stats
    stats = results.get('solution_stats', {})
    if stats:
        print(f"\nSolution Statistics:")
        print(f"  u range:         [{stats.get('u_range', [0,0])[0]:.6e}, {stats.get('u_range', [0,0])[1]:.6e}]")
        print(f"  u mean:          {stats.get('u_mean', 0):.6e}")
        print(f"  u std:           {stats.get('u_std', 0):.6e}")
    
    print("\n" + "=" * 70)
    print("Memory tracking test completed successfully!")
    print("=" * 70)
    
    return output


if __name__ == "__main__":
    main()
