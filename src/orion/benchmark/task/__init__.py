"""
Benchmark Tasks definition
===========================
"""

from .base import BenchmarkTask, bench_task_factory
from .branin import Branin
from .carromtable import CarromTable
from .eggholder import EggHolder
from .forrester import Forrester
from .rosenbrock import RosenBrock

__all__ = [
    "BenchmarkTask",
    "RosenBrock",
    "Branin",
    "CarromTable",
    "EggHolder",
    "Forrester",
    "bench_task_factory",
]
