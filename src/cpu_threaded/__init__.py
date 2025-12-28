"""
CPU Threaded FEM Solver Package

Provides ThreadPoolExecutor-based parallel implementation for Quad-8 FEM computations.
Demonstrates explicit Python threading without JIT compilation.

Note: Due to Python's GIL, speedup is limited to NumPy operations that release
the GIL internally.
"""

from .quad8_cpu_threaded import Quad8FEMSolverThreaded

__all__ = ['Quad8FEMSolverThreaded']
