#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Average Rank Assessment
========================
"""
from collections import defaultdict

from orion.benchmark.assessment.base import BaseAssess
from orion.plotting.base import regrets


class AverageResult(BaseAssess):
    """
    Evaluate the average performance (objective value) for each search algorithm
    at different time steps (trial number).
    The performance (objective value) used for a trial will the best result until the trial.
    """

    def __init__(self, task_num=1):
        super(AverageResult, self).__init__(task_num=task_num)

    def analysis(self, task, experiments):
        """
        Generate a `plotly.graph_objects.Figure` to display average performance
        for each search algorithm.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to run for
            this assessment, and experiment is an instance of `orion.core.worker.experiment`.
        """
        algorithm_groups = defaultdict(list)

        for _, exp in experiments:
            algorithm_name = list(exp.configuration["algorithms"].keys())[0]
            algorithm_groups[algorithm_name].append(exp)

        return regrets(algorithm_groups)
