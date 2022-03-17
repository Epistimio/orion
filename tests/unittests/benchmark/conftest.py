#!/usr/bin/env python
"""Common fixtures and utils for benchmark unit tests."""

import pytest

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.task import CarromTable, RosenBrock


@pytest.fixture()
def task_number():
    """Return task number for assessment"""
    return 2


@pytest.fixture()
def max_trial():
    """Return max trials for task"""
    return 3


@pytest.fixture
def benchmark_algorithms():
    """Return a list of algorithms suitable for Orion experiment"""
    return [{"random": {"seed": 1}}, {"tpe": {"seed": 1}}]


@pytest.fixture
def benchmark_config(benchmark_algorithms):
    config = {
        "name": "bm00001",
        "algorithms": benchmark_algorithms,
        "targets": [
            {
                "assess": {
                    "AverageResult": {"repetitions": 2},
                    "AverageRank": {"repetitions": 2},
                },
                "task": {
                    "RosenBrock": {"dim": 3, "max_trials": 25},
                    "CarromTable": {"max_trials": 20},
                },
            }
        ],
    }

    return config


@pytest.fixture
def benchmark_config_py(benchmark_algorithms):
    config = dict(
        name="bm00001",
        algorithms=benchmark_algorithms,
        targets=[
            {
                "assess": [AverageResult(2), AverageRank(2)],
                "task": [RosenBrock(25, dim=3), CarromTable(20)],
            }
        ],
    )
    return config


@pytest.fixture()
def experiment_config(max_trial, benchmark_algorithms):
    """Return a experiment template configure"""
    config = dict(
        name="experiment-name",
        space={"x": "uniform(0, 200)"},
        metadata={
            "user": "test-user",
            "orion_version": "XYZ",
            "VCS": {
                "type": "git",
                "is_dirty": False,
                "HEAD_sha": "test",
                "active_branch": None,
                "diff_sha": "diff",
            },
        },
        version=1,
        pool_size=1,
        max_trials=max_trial,
        working_dir="",
        algorithms=benchmark_algorithms[0],
        producer={"strategy": "NoParallelStrategy"},
    )
    return config


@pytest.fixture()
def trial_config():
    """Return a trial template configure"""
    trial_config = {
        "experiment": 0,
        "status": "completed",
        "worker": None,
        "start_time": None,
        "end_time": None,
        "heartbeat": None,
        "results": [],
        "params": [],
    }
    return trial_config


@pytest.fixture
def study_experiments_config(
    experiment_config, trial_config, benchmark_algorithms, task_number, max_trial
):
    config = dict(
        exp_config=experiment_config,
        trial_config=trial_config,
        algorithms=benchmark_algorithms,
        task_number=task_number,
        max_trial=max_trial,
    )

    return config
