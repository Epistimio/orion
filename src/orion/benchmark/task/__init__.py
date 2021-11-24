"""
Benchmark Tasks definition
===========================
"""

from .base import BenchmarkTask, bench_task_factory
from .branin import Branin
from .carromtable import CarromTable
from .eggholder import EggHolder
from .rosenbrock import RosenBrock

__all__ = [
    "BenchmarkTask",
    "RosenBrock",
    "Branin",
    "CarromTable",
    "EggHolder",
    "bench_task_factory",
]
