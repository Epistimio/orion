# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.base` -- Base interface and wrapper for a data analyser
=================================================================================

.. module:: base
   :platform: Unix
   :synopsis: Formulation of a general interface for a data analyser to provide
   information to a plotter object

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import (Concept, Wrapper)

log = logging.getLogger(__name__)


class BaseAnalyser(Concept, metaclass=ABCMeta):
    """The base class for analyser defines the main API of a data analyser.
    Every data analyser needs to implement the `analyse` method to return a
    :mod:`orion.viz.analysis.Analysis` object. The type of analysis provided (and thus, returned)
    by the analyser need to be a list of subclasses of the :mod:`orion.viz.analysis.Analysis` class.
    """

    name = "Analyser"

    def __init__(self, trials, experiment, **kwargs):
        """Create an analyser with all the necessary tools for it to do its thing.

        Parameters
        ----------
        trials : List of :mod:`orion.core.worker.trial.Trial`
            All the trials to be analyse.
        experiment : :mod:`orion.core.worker.experiment.Experiment`
            Current experiment being analyzed.
        kwargs : `dict`
            Tunable elements of a particular data analyser.

        """
        self._trials = trials
        self._experiment = experiment
        self._space = experiment.space
        super(BaseAnalyser, self).__init__(trials, experiment, **kwargs)

    @abstractmethod
    def analyse(self, of_type=None):
        """Return a :mod:`orion.viz.analysis.Analysis` object containing the results of the analyse.

        Parameters
        ----------
        of_type : Subclass of :mod:`orion.viz.analysis.Analysis`
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
        """Return the types of analysis this analyser provides."""
        return []


class AnalyserWrapper(Wrapper):
    """Basic wrapper for analysers"""

    implementation_module = "orion.viz.analysers"

    def __init__(self, trials, experiment, analyser_config):
        """Forward the initialization to `Wrapper`"""
        super(AnalyserWrapper, self).__init__(trials, experiment, instance=analyser_config)

    @property
    def wraps(self):
        """Wrap `orion.viz.analysers.base.BaseAnalyser"""
        return BaseAnalyser

    def analyse(self, of_type=None):
        """Wrap the analyse function call by asserting that the requested type is valid."""
        if of_type is not None:
            assert type(of_type) in self.instance.available_analysis

        self.instance.analyse(of_type)
