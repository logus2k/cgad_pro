"""
Unified wrapper for CPU, GPU, Numba, and Numba CUDA solvers.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add solver directories to path
# solver_wrapper.py is in /src/app/server/
# Solver modules are in /src/cpu/, /src/gpu/, /src/numba/, /src/numba_cuda/
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent.parent  # /src/app/server -> /src

sys.path.insert(0, str(SRC_DIR / "cpu"))
sys.path.insert(0, str(SRC_DIR / "gpu"))
sys.path.insert(0, str(SRC_DIR / "numba"))
sys.path.insert(0, str(SRC_DIR / "numba_cuda"))

from quad8_cpu_v3 import Quad8FEMSolver as CPUSolver
from quad8_gpu_v3 import Quad8FEMSolverGPU as GPUSolver
from quad8_numba import Quad8FEMSolverNumba as NumbaSolver
from quad8_numba_cuda import Quad8FEMSolverNumbaCUDA as NumbaCUDASolver


class SolverWrapper:
    """Unified interface for CPU, GPU, Numba, and Numba CUDA solvers"""
    
    def __init__(self, solver_type: str, params: dict, progress_callback=None):
        self.solver_type = solver_type
        self.params = params
        self.progress_callback = progress_callback
        
        # Auto-detect best available solver
        if solver_type == "auto":
            try:
                import cupy
                self.solver_type = "gpu"
            except ImportError:
                self.solver_type = "cpu"
        
        # Create solver instance based on type
        if self.solver_type == "gpu":
            self.solver = GPUSolver(
                mesh_file=params['mesh_file'],
                maxiter=params.get('max_iterations', 15000),
                cg_print_every=params.get('progress_interval', 50),
                verbose=params.get('verbose', True),
                progress_callback=progress_callback
            )
        elif self.solver_type == "numba":
            self.solver = NumbaSolver(
                mesh_file=params['mesh_file'],
                maxiter=params.get('max_iterations', 15000),
                cg_print_every=params.get('progress_interval', 50),
                verbose=params.get('verbose', True),
                progress_callback=progress_callback
            )
        elif self.solver_type == "numba_cuda":
            self.solver = NumbaCUDASolver(
                mesh_file=params['mesh_file'],
                maxiter=params.get('max_iterations', 15000),
                cg_print_every=params.get('progress_interval', 50),
                verbose=params.get('verbose', True),
                progress_callback=progress_callback
            )
        else:
            # Default to CPU
            self.solver = CPUSolver(
                mesh_file=params['mesh_file'],
                maxiter=params.get('max_iterations', 15000),
                cg_print_every=params.get('progress_interval', 50),
                verbose=params.get('verbose', True),
                progress_callback=progress_callback
            )
    
    def run(self) -> Dict[str, Any]:
        """Execute solver and return results"""
        try:
            results = self.solver.run(output_dir=None, export_file=None)
            return results
        except Exception as e:
            if self.progress_callback:
                self.progress_callback.on_error('solve', str(e))
            raise
    
    @staticmethod
    def get_available_solvers() -> list:
        """Return list of available solver types"""
        available = ["cpu"]
        
        try:
            import numba
            available.append("numba")
        except ImportError:
            pass
        
        try:
            import cupy
            available.append("gpu")
        except ImportError:
            pass
        
        try:
            from numba import cuda
            if cuda.is_available():
                available.append("numba_cuda")
        except ImportError:
            pass
        
        return available
