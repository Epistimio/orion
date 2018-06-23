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


class BaseAnalyser(object, metaclass=ABCMeta):
    """Base class describing what a data analyzer can do."""

    def __init__(self, trials, experiment, **kwargs):
        log.debug("Creating Algorithm object of %s type with parameters:\n%s",
                  type(self).__name__, kwargs)

        self._trials = trials
        self._experiment = experiment
        self._space = experiment.space
        self._param_names = list(kwargs.keys())

        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            # Check if tunable element is another algorithm
            if isinstance(param, dict) and len(param) == 1:
                subanalyser_type = list(param)[0]
                subanalyser_kwargs = param[subanalyser_type]
                if isinstance(subanalyser_kwargs, dict):
                    param = AnalyserFactory(subanalyser_type,
                                            trials, experiment, **subanalyser_kwargs)
            elif isinstance(param, str) and \
                    param.lower() in AnalyserFactory.typenames:
                # pylint: disable=too-many-function-args
                param = AnalyserFactory(param, trials, experiment)

            setattr(self, varname, param)

    @abstractmethod
    def analyse(self, of_type=None):
        pass

    @property
    def space(self):
        return self._space

    @property
    def experiment(self):
        return self._experiment

    @property
    def trials(self):
        return self._trials

    @property
    def available_analysis(self):
        return []


# pylint: disable=too-few-public-methods,abstract-method
class AnalyserFactory(BaseAnalyser, metaclass=Factory):
    """Class used to inject dependency on a data analyser implementation.

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseAnalyser` interface.
    """

    pass


class AnalyserWrapper(BaseAnalyser):
    def __init__(self, trials, experiment, analyser_config):
        self.analyser = None
        super(AnalyserWrapper, self).__init__(trials, experiment, analyser=analyser_config)

    def analyse(self, of_type=None):
        return self.analyser.analyse(of_type)

    @property
    def available_analysis(self):
        return self.analyser.available_analysis
