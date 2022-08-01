#!/usr/bin/env python
"""Perform a functional test for orion benchmark."""

import plotly
import pytest

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import BenchmarkTask, Branin
from orion.storage.base import setup_storage

algorithms = [
    {"algorithm": {"random": {"seed": 1}}},
    {"algorithm": {"tpe": {"seed": 1}}},
]


def assert_benchmark_figures(figures, num, assessments, tasks):

    figure_num = 0
    for i, (assess, task_figs) in enumerate(figures.items()):
        assert assess == assessments[i].__class__.__name__
        for j, (task, figs) in enumerate(task_figs.items()):
            assert task == tasks[j].__class__.__name__
            figure_num += len(figs)

            for _, fig in figs.items():
                assert type(fig) is plotly.graph_objects.Figure

    assert figure_num == num


class BirdLike(BenchmarkTask):
    """User defined benchmark task"""

    def __init__(self, max_trials=20):
        super().__init__(max_trials=max_trials)

    def call(self, x):

        y = (2 * x**4 + x**2 + 2) / (x**4 + 1)

        return [dict(name="birdlike", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(-4, 4)"}

        return rspace


@pytest.mark.usefixtures("orionstate")
def test_simple():
    """Test a end 2 end exucution of benchmark"""
    task_num = 2
    max_trials = 10
    assessments = [AverageResult(task_num), AverageRank(task_num)]
    tasks = [
        Branin(max_trials),
        BirdLike(max_trials),
    ]

    storage = setup_storage()
    benchmark = get_or_create_benchmark(
        storage,
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

    assert_benchmark_figures(figures, 4, assessments, tasks)

    storage = setup_storage()
    benchmark = get_or_create_benchmark(storage, name="bm001")
    figures = benchmark.analysis()

    assert_benchmark_figures(figures, 4, assessments, tasks)
    benchmark.close()
