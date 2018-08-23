# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.base` -- Define the base concept and wrapper for plotters
==================================================================================

.. module:: base
   :platform: Unix
   :synopsis: Formulation of a general plotter for visualization

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import Concept

log = logging.getLogger(__name__)


class BasePlotter(Concept, metaclass=ABCMeta):
    """Base class describing what a plotter can do."""

    name = "Plotter"

    def __init__(self, analysis, save_formats, **kwargs):
        self._analysis = analysis
        self.save_formats = save_formats
        super(BasePlotter, self).__init__(analysis, save_formats, **kwargs)

    @abstractmethod
    def plot(self):
        """Plot the analysis"""
        pass

    @property
    def analysis(self):
        """Return the current analysis for this plotter"""
        return self._analysis

    @property
    def required_analysis(self):
        """Return the analysis that this plotter requires to work"""
        return []
