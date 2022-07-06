#!/usr/bin/env python
"""
Parallel Advantage Assessment
=============================
"""

from collections import defaultdict

from orion.benchmark.assessment.base import BenchmarkAssessment
from orion.executor.base import executor_factory
from orion.plotting.base import durations, parallel_assessment, regrets


class ParallelAssessment(BenchmarkAssessment):
    """
    Evaluate how algorithms' sampling efficiency is affected by different degrees of parallelization.

    Evaluate the average performance (objective value) for each search algorithm
    at different time steps (trial number).
    The performance (objective value) used for a trial will the best result until the trial.

    Parameters
    ----------
    task_num: int, optional
        Number of experiment to run for each number of workers. Default: 1
    executor: str, optional
        Name of orion worker exeuctor. If `None`, the default executor of the benchmark will be used. Default: `None`.
    n_workers: list or tuple, optional
        List or tuple of integers for the number of workers for each experiment. Default: (1, 2, 4)
    **executor_config: dict
        Parameters for the corresponding executor.
    """

    def __init__(
        self, task_num=1, executor=None, n_workers=(1, 2, 4), **executor_config
    ):

        super().__init__(
            task_num=task_num * len(n_workers),
            executor=executor,
            n_workers=n_workers,
            **executor_config
        )
        self.worker_num = len(n_workers)
        self.executor_name = executor
        self.executor_config = executor_config
        self.workers = [n_worker for n_worker in n_workers for _ in range(task_num)]

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

            n_worker = self.workers[task_index]
            algo_key = algorithm_name + "_workers_" + str(n_worker)
            algorithm_worker_groups[algo_key].append(exp)

        assessment = self.__class__.__name__

        figure = defaultdict(dict)
        figure[assessment][task] = dict()

        figure[assessment][task][parallel_assessment.__name__] = parallel_assessment(
            algorithm_groups
        )
        figure[assessment][task][durations.__name__] = durations(
            algorithm_worker_groups
        )
        figure[assessment][task][regrets.__name__] = regrets(algorithm_worker_groups)

        return figure

    def get_executor(self, task_index):
        return executor_factory.create(
            self.executor_name,
            n_workers=self.workers[task_index],
            **self.executor_config
        )

    @property
    def configuration(self):
        """Return the configuration of the assessment."""
        config = super().configuration
        config[self.__class__.__qualname__]["task_num"] = int(
            self.task_num / self.worker_num
        )
        return config
