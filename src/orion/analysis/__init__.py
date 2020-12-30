# -*- coding: utf-8 -*-
"""
:mod:`orion.analysis` -- Provides HPO analysis tools
====================================================

.. module:: analysis
   :platform: Unix
   :synopsis: Provides agnostic HPO analysis tools
"""

from orion.analysis.lpi import lpi
from orion.analysis.regret import regret

__all__ = ["lpi", "regret"]
