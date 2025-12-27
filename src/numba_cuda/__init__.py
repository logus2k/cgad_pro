"""
Numba CUDA FEM Solver Package

Provides @cuda.jit GPU kernel implementations for Quad-8 FEM computations
as an alternative to CuPy RawKernel (CUDA C).

Demonstrates Python-syntax GPU programming with Numba.
"""

from .quad8_numba_cuda import Quad8FEMSolverNumbaCUDA
from .kernels_numba_cuda import (
    quad8_assembly_kernel,
    quad8_postprocess_kernel,
    get_gauss_points_9,
    get_gauss_points_4
)

__all__ = [
    'Quad8FEMSolverNumbaCUDA',
    'quad8_assembly_kernel',
    'quad8_postprocess_kernel',
    'get_gauss_points_9',
    'get_gauss_points_4',
]
