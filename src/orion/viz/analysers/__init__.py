# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers` -- Base interface, factory and wrapper for DataAdapter
=================================================================================

.. module:: analysers
   :platform: Unix
   :synopsis: Formulation of a general interface for a data analyser to provide
   information to a plotter object

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import (Factory, get_qualified_name)

log = logging.getLogger(__name__)


class BaseAnalyser(object, metaclass=ABCMeta):
    """Base class describing what a data analyzer can do."""

    def __init__(self, trials, experiment, **kwargs):
        """Create an analyser with all the necessary tools for it to do its thing.

        Parameters
        ----------
        trials : List of `orion.core.worker.trial.Trial`
            All the trials to be analyse.
        experiment : `orion.core.worker.experiment.Experiment`
            Current experiment being analyzed.
        kwargs : dict
            Tunable elements of a particular data analyser.

        """
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
                    try:
                        qualified_name = get_qualified_name(self.__module__, subanalyser_type)
                        param = AnalyserFactory((qualified_name, subanalyser_type),
                                                trials, experiment, **subanalyser_kwargs)
                    except NotImplementedError:
                        pass
            elif isinstance(param, str) and \
                    get_qualified_name(get_qualified_name(self.__module__, param),
                                       param) in AnalyserFactory.typenames:
                # pylint: disable=too-many-function-args
                param = AnalyserFactory((get_qualified_name(self.__module__, param),
                                        param), trials, experiment)

            setattr(self, varname, param)

    @abstractmethod
    def analyse(self, of_type=None):
        """Return a `orion.viz.analysis.Analysis` object containing the results of the analyse.

        Parameters
        ----------
        of_type : Subclass of `orion.viz.analysis.Analysis`
            If the data analyser provides different type of analyses, this tells the
            object which one we want.
        """
        pass

    @property
    def space(self):
        """Return problem space."""
        return self._space

    @property
    def experiment(self):
        """Return current experiment."""
        return self._experiment

    @property
    def trials(self):
        """Return all trials."""
        return self._trials

    @property
    def available_analysis(self):
        """Return the type of analysis this analyser provides."""
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
