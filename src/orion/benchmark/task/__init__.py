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
