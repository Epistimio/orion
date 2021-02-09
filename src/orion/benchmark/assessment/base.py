#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base definition of Assessment
==============================
"""

from abc import ABC, abstractmethod

from orion.core.utils import Factory


class BaseAssess(ABC):
    """Base class describing what an assessment can do.

    Parameters
    ----------
    task_num : int
       Number of experiment the assessment ask to run the corresponding task
    kwargs : dict
       Configurable parameters of the assessment, a particular assessment
       implementation can have its own parameters.
    """

    def __init__(self, task_num, **kwargs):
        self.task_number = task_num
        self._param_names = kwargs
        self._param_names["task_num"] = task_num

    @property
    def task_num(self):
        """Return the task number to run for this assessment"""
        return self.task_number

    @abstractmethod
    def analysis(self, task, experiments):
        """
        Generate a `plotly.graph_objects.Figure` to display the performance analysis
        based on the assessment purpose.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to run for
            this assessment, and experiment is an instance of `orion.core.worker.experiment`.
        """
        pass

    @property
    def configuration(self):
        """Return the configuration of the assessment."""
        return {self.__class__.__qualname__: self._param_names}


# pylint: disable=too-few-public-methods,abstract-method
class BenchmarkAssessment(BaseAssess, metaclass=Factory):
    """Class used to inject dependency on an assessment implementation."""

    pass
