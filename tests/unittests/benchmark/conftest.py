#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for benchmark unit tests."""

import datetime

import pytest

from orion.testing import generate_trials


@pytest.fixture()
def task_number():
    """Return task number for assessment"""
    return 2


@pytest.fixture()
def max_trial():
    """Return max trials for task"""
    return 3


@pytest.fixture
def algorithms():
    """Return a list of algorithms suitable for Orion experiment"""
    return [{"random": {"seed": 1}}, {"tpe": {"seed": 1}}]


@pytest.fixture()
def experiment_config(max_trial, algorithms):
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
        algorithms=algorithms[0],
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
def generate_experiment_trials(
    algorithms, experiment_config, trial_config, task_number, max_trial
):
    """Return a list of experiments and trials"""
    gen_exps = []
    gen_trials = []
    algo_num = len(algorithms)
    for i in range(task_number * algo_num):
        import copy

        exp = copy.deepcopy(experiment_config)
        exp["_id"] = i
        exp["name"] = "experiment-name-{}".format(i)
        exp["algorithms"] = algorithms[i % algo_num]
        exp["max_trials"] = max_trial
        exp["metadata"]["datetime"] = datetime.datetime.utcnow()
        gen_exps.append(exp)
        for j in range(max_trial):
            trial = copy.deepcopy(trial_config)
            trial["_id"] = "{}{}".format(i, j)
            trial["experiment"] = i
            trials = generate_trials(trial, ["completed"])
            gen_trials.extend(trials)

    return gen_exps, gen_trials
