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
    """Plotter objects receive an :mod:`orion.viz.analysis.Analysis` object and use it
    to create a visual representation of the data stored inside the analysis. This can
    be as simple as output it to the console or creating intricate graphs using dedicated
    plotting libaries. They provide this functionality through the `plot` method. They also
    need to indicate which analysis they can work on through the `required_analysis` property.
    """

    name = "Plotter"

    def __init__(self, analysis, save_formats, **kwargs):
        """Call the base class."""
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
