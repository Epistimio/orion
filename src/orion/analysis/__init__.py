# -*- coding: utf-8 -*-
"""
:mod:`orion.analysis` -- Provides HPO analysis tools
====================================================

.. module:: analysis
   :platform: Unix
   :synopsis: Provides agnostic HPO analysis tools
"""

from orion.analysis.lpi_utils import lpi
from orion.analysis.partial_dependency_utils import partial_dependency
from orion.analysis.regret_utils import regret

__all__ = ["lpi", "partial_dependency", "regret"]
