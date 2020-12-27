"""
:mod:`orion.benchmark.assessment` -- Benchmark Assessments definition
=======================================================================

.. module:: assessment
   :platform: Unix
   :synopsis: Benchmark Assessments definition.

"""

from .averagerank import AverageRank
from .averageresult import AverageResult

__all__ = ['AverageRank', 'AverageResult']
