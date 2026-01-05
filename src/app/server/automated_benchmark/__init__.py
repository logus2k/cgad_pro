"""
Automated Benchmark Suite for FEMulator Pro.

Provides automated benchmark testing across all solver implementations.
"""

from .config_loader import ConfigLoader, TestCase
from .result_recorder import ResultRecorder
from .runner import BenchmarkRunner, ProgressTracker

__all__ = [
    'ConfigLoader',
    'TestCase',
    'ResultRecorder',
    'BenchmarkRunner',
    'ProgressTracker'
]
