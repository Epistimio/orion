#!/usr/bin/env python
"""
Base definition of Assessment
==============================
"""

from abc import ABC, abstractmethod

from orion.core.utils import GenericFactory


class BenchmarkAssessment(ABC):
    """Base class describing what an assessment can do.

    Parameters
    ----------
    repetitions : int
       Number of experiment the assessment ask to run the corresponding task
    kwargs : dict
       Configurable parameters of the assessment, a particular assessment
       implementation can have its own parameters.
    """

    def __init__(self, repetitions, **kwargs):
        self._repetitions = repetitions
        self._param_names = kwargs

        self._param_names["repetitions"] = repetitions

    @property
    def repetitions(self):
        """Return the task number to run for this assessment"""
        return self._repetitions

    def get_executor(self, task_index):
        """Return an instance of `orion.executor.base.Executor` based on the index of tasks
        that the assessment is asking to run."""
        return None

    @abstractmethod
    def analysis(self, task, experiments):
        """
        Generate `plotly.graph_objects.Figure` objects to display the performance analysis
        based on the assessment purpose.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to run for
            this assessment, and experiment is an instance of `orion.core.worker.experiment`.

        Returns
        -------
        Dict of plotly.graph_objects.Figure objects with a format as like
        {"assessment name": {"task name": {"figure name": plotly.graph_objects.Figure}}}

        Examples
        >>> {"AverageRank": {"RosenBrock": {"rankings": plotly.graph_objects.Figure}}}

        """

    @property
    def configuration(self):
        """Return the configuration of the assessment."""
        return {self.__class__.__qualname__: self._param_names}


bench_assessment_factory = GenericFactory(BenchmarkAssessment)
