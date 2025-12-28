"""
CPU Multiprocess FEM Solver Package

Provides multiprocessing.Pool-based parallel implementation for Quad-8 FEM computations.
Demonstrates explicit Python multiprocessing - true parallelism bypassing GIL.
"""

from .quad8_cpu_multiprocess import Quad8FEMSolverMultiprocess

__all__ = ['Quad8FEMSolverMultiprocess']
