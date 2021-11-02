"""
Benchmark Assessments definition
=================================
"""

from .averagerank import AverageRank
from .averageresult import AverageResult
from .base import BaseAssess
from .paralleladvantage import ParallelAdvantage

__all__ = ["BaseAssess", "AverageRank", "AverageResult", "ParallelAdvantage"]
