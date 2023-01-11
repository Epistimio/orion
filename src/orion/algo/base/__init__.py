"""Base module to create new algorithm a instantiate them"""

from .base import BaseAlgorithm, algo_factory
from .registry import Registry

__all__ = ["BaseAlgorithm", "algo_factory", "Registry"]
