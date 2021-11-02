#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Advantage Assessment
=============================
"""

from collections import defaultdict

from orion.benchmark.assessment.base import BaseAssess
from orion.executor.base import Executor
from orion.plotting.base import parallel_advantage


class ParallelAdvantage(BaseAssess):
    """
    Evaluate the advantage to run experiment in multiple parallel workers.

    Evaluate the average performance (objective value) for each search algorithm
    at different time steps (trial number).
    The performance (objective value) used for a trial will the best result until the trial.
    """

    def __init__(
        self, task_num=1, executor=None, n_workers=[1, 2, 4], **executor_config
    ):
        if task_num != 1:
            raise ValueError("ParallelAdvantage only supports task_num=1")

        super(ParallelAdvantage, self).__init__(
            task_num=task_num * len(n_workers),
            executor=executor,
            n_workers=n_workers,
            **executor_config
        )
        self.executor_name = executor
        self.executor_config = executor_config
        self.n_workers = [n_worker for n_worker in n_workers for _ in range(task_num)]

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
            # n_worker = self.n_workers[task_index]
            algorithm_name = list(exp.configuration["algorithms"].keys())[0]
            algorithm_groups[algorithm_name].append(exp)

        return parallel_advantage(algorithm_groups)
        # return self._viz_parallel(algorithm_groups)

    def executor(self, task_index):
        return Executor(
            self.executor_name,
            n_workers=self.n_workers[task_index],
            **self.executor_config
        )
