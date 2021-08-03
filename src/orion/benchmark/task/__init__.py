"""
Benchmark Tasks definition
===========================
"""

from .base import BaseTask
from .branin import Branin
from .carromtable import CarromTable
from .eggholder import EggHolder
from .rosenbrock import RosenBrock

__all__ = ["BaseTask", "RosenBrock", "Branin", "CarromTable", "EggHolder"]

# IDEA: Add an external entry_point for Tasks, so that we can add the profet and OpenML
# tasks as an optional extras_require.
