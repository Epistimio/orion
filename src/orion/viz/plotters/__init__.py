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

from orion.core.utils import Factory

log = logging.getLogger(__name__)


class BasePlotter(object, metaclass=ABCMeta):
    """Base class describing what a data analyzer can do."""

    def __init__(self, analysis, save_formats, **kwargs):
        log.debug("Creating Algorithm object of %s type with parameters:\n%s",
                  type(self).__name__, kwargs)

        self._analysis = analysis
        self._param_names = list(kwargs.keys())

        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            # Check if tunable element is another algorithm
            if isinstance(param, dict) and len(param) == 1:
                subplotter_type = list(param)[0]
                subplotter_kwargs = param[subplotter_type]
                if isinstance(subplotter_kwargs, dict):
                    param = PlotterFactory(subplotter_type,
                                           analysis, save_formats, **subplotter_kwargs)
            elif isinstance(param, str) and \
                    param.lower() in PlotterFactory.typenames:
                # pylint: disable=too-many-function-args
                param = PlotterFactory(param, analysis, save_formats)

            setattr(self, varname, param)

    @abstractmethod
    def plot(self):
        pass

    @property
    def analysis(self):
        return self._analysis

    @property
    def required_analysis(self):
        return []


# pylint: disable=too-few-public-methods,abstract-method
class PlotterFactory(BasePlotter, metaclass=Factory):
    """Class used to inject dependency on a plotter implementation.

    .. seealso:: `orion.core.utils.Factory` metaclass and `BasePlotter` interface.
    """

    pass


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
