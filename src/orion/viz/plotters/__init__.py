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

from orion.core.utils import (Factory, get_qualified_name)

log = logging.getLogger(__name__)


class BasePlotter(object, metaclass=ABCMeta):
    """Base class describing what a data analyzer can do."""

    def __init__(self, analysis, save_formats, module_addendum, **kwargs):
        log.debug("Creating Algorithm object of %s type with parameters:\n%s",
                  type(self).__name__, kwargs)

        self._analysis = analysis
        self._param_names = list(kwargs.keys())

        # TODO change this addendum to support multi-level instances
        use_module = self.__module__

        if module_addendum != '' and not use_module.endswith(module_addendum):
            use_module += '.' + module_addendum

        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            # Check if tunable element is another algorithm
            if isinstance(param, dict) and len(param) > 0:
                try:
                    subplotter_type = list(param)[0]
                    subplotter_kwargs = param[subplotter_type]
                    if isinstance(subplotter_kwargs, dict):
                        qualified_name = get_qualified_name(use_module, subplotter_type)
                        param = PlotterFactory((qualified_name, subplotter_type),
                                               analysis, save_formats, **subplotter_kwargs)

                    if isinstance(param, dict) and len(param) > 1:
                        for subvar, subparam in param.items()[1:]:
                            setattr(self, subvar, subparam)

                except NotImplementedError:
                    pass
            elif isinstance(param, str) and \
                    get_qualified_name(get_qualified_name(use_module, param),
                                       param) in PlotterFactory.typenames:
                # pylint: disable=too-many-function-args
                param = PlotterFactory((get_qualified_name(use_module, param),
                                       param), analysis, save_formats)

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
    def __init__(self, analysis, save_formats, plotter_config, module_addendum=''):
        self.plotter = None
        super(PlotterWrapper, self).__init__(analysis, save_formats, module_addendum,
                                             plotter=plotter_config)

        if type(analysis) not in self.required_analysis:
            raise TypeError('Analysis type not supported by this plotter')

    def plot(self):
        return self.plotter.plot()

    @property
    def required_analysis(self):
        return self.plotter.required_analysis
