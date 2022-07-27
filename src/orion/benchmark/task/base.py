#!/usr/bin/env python
"""
Base definition of Task
========================
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from orion.core.utils import GenericFactory


class BenchmarkTask(ABC):
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

    def __init__(self, max_trials: int, **kwargs):
        self._max_trials = max_trials
        self._param_names = kwargs
        self._param_names["max_trials"] = max_trials

    @abstractmethod
    def call(self) -> List[Dict]:
        """
        Define the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.

        This method should be overridden by subclasses. It should receive the hyper-parameters
        as keyword arguments, with argument names matching the keys of the dictionary returned by
        `get_search_space`.
        """

    def __call__(self, *args, **kwargs):
        """
        All tasks will be callable by default, and method `call()` will be executed when a task is
        called directly.
        """
        return self.call(*args, **kwargs)

    @property
    def max_trials(self) -> int:
        """Return the max number of trials to run for the task."""
        return self._max_trials

    @abstractmethod
    def get_search_space(self) -> Dict[str, str]:
        """Return the search space for the task objective function"""

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task."""
        return {self.__class__.__qualname__: self._param_names}


bench_task_factory = GenericFactory(BenchmarkTask)
