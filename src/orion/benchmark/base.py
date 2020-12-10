#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.base` -- Base definition of Task and Assessment
======================================================================

.. module:: base
   :platform: Unix
   :synopsis: Base definition of Task and Assessment.

"""

from abc import ABC, abstractmethod


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
        self._param_names = list(kwargs.keys())

    @property
    def task_num(self):
        """Return the task number to run for this assessment"""
        return self.task_number

    @abstractmethod
    def plot_figures(self, task, experiments):
        """
        Generate a `plotly.graph_objects.Figure`

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
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            dict_form[attrname] = attr
        dict_form['task_num'] = self.task_num

        mod = self.__class__.__module__
        fullname = mod + '.' + self.__class__.__qualname__
        fullname = fullname.replace('.', '-')
        return {fullname: dict_form}


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
        """
        - build orion experiment
        """
        self.trials_num = max_trials
        self._param_names = list(kwargs.keys())

    @abstractmethod
    def get_blackbox_function(self):
        """
        Return the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """
        pass

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
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            dict_form[attrname] = attr
        dict_form['max_trials'] = self.max_trials

        mod = self.__class__.__module__
        fullname = mod + '.' + self.__class__.__qualname__
        fullname = fullname.replace('.', '-')
        return {fullname: dict_form}
