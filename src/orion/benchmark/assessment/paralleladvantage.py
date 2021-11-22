#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Advantage Assessment
=============================
"""

from collections import defaultdict

from orion.benchmark.assessment.base import BenchmarkAssessment
from orion.executor.base import executor_factory
from orion.plotting.base import durations, parallel_advantage, regrets


class ParallelAdvantage(BenchmarkAssessment):
    """
    Evaluate the advantage to run experiment in multiple parallel workers.

    Evaluate the average performance (objective value) for each search algorithm
    at different time steps (trial number).
    The performance (objective value) used for a trial will the best result until the trial.

    Parameters
    ----------
    task_num: int
        Number of experiment to run for each number of workers.
    executor: str
        Name of orion worker exeuctor.
    n_workers: list
        List of intergers for number of workers for each experiment.
    executor_config: dict
        Parameters for the corresponding executor.
    """

    def __init__(
        self, task_num=1, executor=None, n_workers=[1, 2, 4], **executor_config
    ):

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
        algorithm_worker_groups = defaultdict(list)
        for task_index, exp in experiments:
            algorithm_name = list(exp.configuration["algorithms"].keys())[0]
            algorithm_groups[algorithm_name].append(exp)

            n_worker = self.n_workers[task_index]
            algo_key = algorithm_name + "_workers_" + str(n_worker)
            algorithm_worker_groups[algo_key].append(exp)

        figures = list()
        figures.append(parallel_advantage(algorithm_groups))
        figures.append(durations(algorithm_worker_groups))
        figures.append(regrets(algorithm_worker_groups))

        return figures

    def executor(self, task_index):
        return executor_factory.create(
            self.executor_name,
            n_workers=self.n_workers[task_index],
            **self.executor_config
        )
