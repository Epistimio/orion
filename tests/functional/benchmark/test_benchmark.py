#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for orion benchmark."""

import plotly

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.base import BaseTask
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import Branin, CarromTable, EggHolder, RosenBrock


algorithms = [{'random': {'seed': 1}}, {'tpe': {'seed': 1}}]


class BirdLike(BaseTask):
    """User defined benchmark task"""

    def __init__(self, max_trials=20):
        super(BirdLike, self).__init__(max_trials=max_trials)

    def get_blackbox_function(self):
        """
        Return the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """
        def birdlike(x):

            y = (2 * x ** 4 + x ** 2 + 2) / (x ** 4 + 1)

            return [dict(
                name='birdlike',
                type='objective',
                value=y)]

        return birdlike

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {'x': 'uniform(-4, 4)'}

        return rspace


def test_simple():
    """Test a end 2 end exucution of benchmark"""
    task_num = 2
    trial_num = 20
    assessments = [AverageResult(task_num), AverageRank(task_num)]
    tasks = [RosenBrock(trial_num, dim=3), EggHolder(trial_num, dim=4),
             CarromTable(trial_num), Branin(trial_num), BirdLike(trial_num)]
    benchmark = get_or_create_benchmark(name='bm001',
                                        algorithms=algorithms,
                                        targets=[{
                                            'assess': assessments,
                                            'task': tasks}])
    benchmark.process()

    assert len(benchmark.studies) == len(assessments) * len(tasks)

    status = benchmark.status()

    experiments = benchmark.experiments()

    assert len(experiments) == len(algorithms) * task_num * len(assessments) * len(tasks)

    assert len(status) == len(algorithms) * len(assessments) * len(tasks)

    figures = benchmark.analysis()

    assert len(figures) == len(benchmark.studies)
    assert type(figures[0]) is plotly.graph_objects.Figure

    benchmark = get_or_create_benchmark(name='bm001')
    figures = benchmark.analysis()

    assert len(figures) == len(benchmark.studies)
    assert type(figures[0]) is plotly.graph_objects.Figure
