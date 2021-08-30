#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base definition of Task
========================
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

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

    def __init__(self, max_trials: int, **kwargs):
        self._max_trials = max_trials
        self._param_names = kwargs
        self._param_names["max_trials"] = max_trials

    @abstractmethod
    def call(self, x) -> List[Dict]:
        """
        Define the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """

    def __call__(self, *args, **kwargs):
        """
        All tasks will be callable by default,
        and method `call()` will be executed when a task is called directly.
        """
        return self.call(*args, **kwargs)

    @property
    def max_trials(self) -> int:
        """Return the max number of trials to run for the task. """
        return self._max_trials

    @abstractmethod
    def get_search_space(self) -> Dict[str, str]:
        """Return the search space for the task objective function"""

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task."""
        return {self.__class__.__qualname__: self._param_names}

    @property
    def unwrapped(self) -> "BaseTask":
        """ Returns the 'unwrapped' task, in this case, `self`. """
        return self


# pylint: disable=too-few-public-methods,abstract-method
class BenchmarkTask(BaseTask, metaclass=Factory):
    """Class used to inject dependency on an task implementation."""
