"""
Benchmark Tasks definition
===========================
"""

from .base import BenchmarkTask, bench_task_factory
from .branin import Branin
from .carromtable import CarromTable
from .eggholder import EggHolder
from .rosenbrock import RosenBrock
from .forrester import Forrester

try:
    from . import profet

    # from .profet import ProfetSvmTask, ProfetFcNetTask, ProfetForresterTask, ProfetXgBoostTask
except ImportError:
    pass

__all__ = [
    "BenchmarkTask",
    "RosenBrock",
    "Branin",
    "CarromTable",
    "EggHolder",
    "Forrester",
    "profet",
    # "ProfetSvmTask",
    # "ProfetFcNetTask",
    # "ProfetForresterTask",
    # "ProfetXgBoostTask",
    "bench_task_factory",
]
