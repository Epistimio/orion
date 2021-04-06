#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for orion benchmark."""

import plotly
import pytest

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import BaseTask, Branin, CarromTable, EggHolder, RosenBrock

algorithms = [
    {"algorithm": {"random": {"seed": 1}}},
    {"algorithm": {"tpe": {"seed": 1}}},
]


class BirdLike(BaseTask):
    """User defined benchmark task"""

    def __init__(self, max_trials=20):
        super(BirdLike, self).__init__(max_trials=max_trials)

    def call(self, x):

        y = (2 * x ** 4 + x ** 2 + 2) / (x ** 4 + 1)

        return [dict(name="birdlike", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(-4, 4)"}

        return rspace


@pytest.mark.usefixtures("setup_pickleddb_database")
def test_simple():
    """Test a end 2 end exucution of benchmark"""
    task_num = 2
    trial_num = 10
    assessments = [AverageResult(task_num), AverageRank(task_num)]
    tasks = [
        Branin(trial_num),
        BirdLike(trial_num),
    ]

    benchmark = get_or_create_benchmark(
        name="bm001",
        algorithms=algorithms,
        targets=[{"assess": assessments, "task": tasks}],
    )

    benchmark.process()

    assert len(benchmark.studies) == len(assessments) * len(tasks)

    status = benchmark.status()

    experiments = benchmark.experiments()

    assert len(experiments) == len(algorithms) * task_num * len(assessments) * len(
        tasks
    )

    assert len(status) == len(algorithms) * len(assessments) * len(tasks)

    figures = benchmark.analysis()

    assert len(figures) == len(benchmark.studies)
    assert type(figures[0]) is plotly.graph_objects.Figure

    benchmark = get_or_create_benchmark(name="bm001")
    figures = benchmark.analysis()

    assert len(figures) == len(benchmark.studies)
    assert type(figures[0]) is plotly.graph_objects.Figure
