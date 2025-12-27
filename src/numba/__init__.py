"""
Numba-optimized FEM Solver Package

Provides JIT-compiled implementations of Quad-8 FEM computations
using Numba for near-native performance without leaving Python.

Key optimizations:
- @njit decorated element functions
- Parallel assembly using prange
- Cached compilation for faster subsequent runs
"""

from .quad8_numba import Quad8FEMSolverNumba
from .elem_quad8_numba import elem_quad8, elem_quad8_velocity
from .shape_n_der8_numba import shape_n_der8
from .robin_quadr_numba import robin_quadr
from .genip2dq_numba import genip2dq_4, genip2dq_9, genip1d_3

__all__ = [
    'Quad8FEMSolverNumba',
    'elem_quad8',
    'elem_quad8_velocity', 
    'shape_n_der8',
    'robin_quadr',
    'genip2dq_4',
    'genip2dq_9',
    'genip1d_3',
]
