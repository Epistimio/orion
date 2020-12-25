"""
:mod:`orion.benchmark.task` -- Benchmark Tasks definition
================================================================

.. module:: assessment
   :platform: Unix
   :synopsis: Benchmark Assessments definition.

"""

from .branin import Branin
from .carromtable import CarromTable
from .eggholder import EggHolder
from .rosenbrock import RosenBrock

__all__ = ['RosenBrock', 'Branin', 'CarromTable', 'EggHolder']
