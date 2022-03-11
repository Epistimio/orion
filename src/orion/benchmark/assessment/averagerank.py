#!/usr/bin/env python
"""
Average Rank Assessment
========================
"""

from collections import defaultdict

from orion.benchmark.assessment.base import BenchmarkAssessment
from orion.plotting.base import rankings


class AverageRank(BenchmarkAssessment):
    """
    Evaluate the average performance (objective value) between different search algorithms from
    the rank perspective at different time steps (trial number).
    The performance (objective value) used for a trial will the best result until the trial.
    """

    def __init__(self, task_num=1):
        super().__init__(task_num=task_num)

    def analysis(self, task, experiments):
        """
        Generate a `plotly.graph_objects.Figure` to display average rankings
        between different search algorithms.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to run for
            this assessment, and experiment is an instance of `orion.core.worker.experiment`.
        """
        algorithm_groups = defaultdict(list)

        for exp in experiments:
            algorithm_name = list(exp.configuration["algorithms"].keys())[0]
            algorithm_groups[algorithm_name].append(exp)

        assessment = self.__class__.__name__
        figures = defaultdict(dict)
        figures[assessment][task] = dict()
        figures[assessment][task][rankings.__name__] = rankings(algorithm_groups)
        return figures
