# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.base` -- What is a data analyzer
==============================================================================

.. module:: base
   :platform: Unix
   :synopsis: Formulation of a general data analyser for visualization

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import Concept

log = logging.getLogger(__name__)


class BasePlotter(Concept, metaclass=ABCMeta):
    """Base class describing what a data analyzer can do."""

    name = "Plotter"

    def __init__(self, analysis, save_formats, **kwargs):
        self._analysis = analysis
        self.save_formats = save_formats

        super(BasePlotter, self).__init__(analysis, save_formats, **kwargs)

    @abstractmethod
    def plot(self):
        pass

    @property
    def analysis(self):
        return self._analysis

    @property
    def required_analysis(self):
        return []


class PlotterWrapper(BasePlotter):
    def __init__(self, analysis, save_formats, plotter_config):
        self.plotter = None
        super(PlotterWrapper, self).__init__(analysis, save_formats, plotter=plotter_config)

        if type(analysis) not in self.required_analysis:
            raise TypeError('Analysis type not supported by this plotter')

    def plot(self):
        return self.plotter.plot()

    @property
    def required_analysis(self):
        return self.plotter.required_analysis
