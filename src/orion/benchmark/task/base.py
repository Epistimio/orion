#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base definition of Task
========================
"""

from abc import ABC, abstractmethod

from orion.core.utils import Factory


class BaseTask(ABC):
    """Base class describing what a task can do.
    A task will define the objective function and search space of it.

    Parameters
    ----------
    max_trials : int
       Max number of trials the experiment will run against this task.
    kwargs : dict
       Configurable parameters of the task, a particular task
       implementation can have its own parameters.
    """

    def __init__(self, max_trials, **kwargs):
        self.trials_num = max_trials
        self._param_names = kwargs
        self._param_names["max_trials"] = max_trials

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        Define the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        All tasks will be callable by default,
        and method `call()` will be executed when a task is called directly.
        """
        return self.call(*args, **kwargs)

    @property
    def max_trials(self):
        """Return the max number of trials to run for the"""
        return self.trials_num

    @abstractmethod
    def get_search_space(self):
        """Return the search space for the task objective function"""
        pass

    @property
    def configuration(self):
        """Return the configuration of the task."""
        return {self.__class__.__qualname__: self._param_names}


# pylint: disable=too-few-public-methods,abstract-method
class BenchmarkTask(BaseTask, metaclass=Factory):
    """Class used to inject dependency on an task implementation."""

    pass
