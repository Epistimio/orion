# -*- coding: utf-8 -*-
"""
Common testing support module
=============================

Common testing support module providing defaults, functions and mocks.

"""
# pylint: disable=protected-access

import contextlib
import copy
import datetime
import os
from contextlib import contextmanager

import orion.algo.space
import orion.core.io.experiment_builder as experiment_builder
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.producer import Producer
from orion.testing.state import OrionState


def default_datetime():
    """Return default datetime"""
    return datetime.datetime(1903, 4, 25, 0, 0, 0)


def generate_trials(trial_config, statuses, exp_config=None):
    """Generate Trials with different configurations"""

    def _generate(obj, *args, value):
        if obj is None:
            return None

        obj = copy.deepcopy(obj)
        data = obj

        data[args[-1]] = value
        return obj

    new_trials = [_generate(trial_config, "status", value=s) for s in statuses]

    for i, trial in enumerate(new_trials):
        trial["submit_time"] = datetime.datetime.utcnow() + datetime.timedelta(
            seconds=i
        )
        if trial["status"] != "new":
            trial["start_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    for i, trial in enumerate(new_trials):
        if trial["status"] == "completed":
            trial["end_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    if exp_config:
        space = SpaceBuilder().build(exp_config["space"])
    else:
        space = SpaceBuilder().build({"x": "uniform(0, 200)"})

    # make each trial unique
    for i, trial in enumerate(new_trials):
        if trial["status"] == "completed":
            trial["results"].append({"name": "loss", "type": "objective", "value": i})

        trial_stub = tuple_to_trial(space.sample(seed=i)[0], space)
        trial["params"] = trial_stub.to_dict()["params"]

    return new_trials


def generate_benchmark_experiments_trials(
    benchmark_algorithms, experiment_config, trial_config, task_number, max_trial
):
    """Return a list of experiments and trials for a benchmark"""
    gen_exps = []
    gen_trials = []
    algo_num = len(benchmark_algorithms)
    for i in range(task_number * algo_num):
        import copy

        exp = copy.deepcopy(experiment_config)
        exp["_id"] = i
        exp["name"] = "experiment-name-{}".format(i)
        exp["algorithms"] = benchmark_algorithms[i % algo_num]["algorithm"]
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


@contextmanager
def create_study_experiments(
    exp_config, trial_config, algorithms, task_number, max_trial
):
    gen_exps, gen_trials = generate_benchmark_experiments_trials(
        algorithms, exp_config, trial_config, task_number, max_trial
    )
    with OrionState(experiments=gen_exps, trials=gen_trials):
        experiments = []
        experiments_info = []
        for i in range(task_number * len(algorithms)):
            experiment = experiment_builder.build("experiment-name-{}".format(i))
            experiments.append(experiment)

        for index, exp in enumerate(experiments):
            experiments_info.append((int(index / task_number), exp))

        yield experiments_info


def mock_space_iterate(monkeypatch):
    """Force space to return seeds as samples instead of actually sampling

    This is useful for tests where we want to get params we can predict (0, 1, 2, ...)
    """
    sample = orion.algo.space.Space.sample

    def iterate(self, seed, *args, **kwargs):
        """Return the points with seed value instead of sampling"""
        points = []
        for point in sample(self, seed=seed, *args, **kwargs):
            points.append([seed] * len(point))
        return points

    monkeypatch.setattr("orion.algo.space.Space.sample", iterate)


@contextmanager
def create_experiment(exp_config=None, trial_config=None, statuses=None):
    """Context manager for the creation of an ExperimentClient and storage init"""
    if exp_config is None:
        raise ValueError("Parameter 'exp_config' is missing")
    if trial_config is None:
        raise ValueError("Parameter 'trial_config' is missing")
    if statuses is None:
        statuses = ["new", "interrupted", "suspended", "reserved", "completed"]

    from orion.client.experiment import ExperimentClient

    with OrionState(
        experiments=[exp_config],
        trials=generate_trials(trial_config, statuses, exp_config),
    ) as cfg:
        experiment = experiment_builder.build(name=exp_config["name"])
        if cfg.trials:
            experiment._id = cfg.trials[0]["experiment"]
        client = ExperimentClient(experiment, Producer(experiment))
        yield cfg, experiment, client

    client.close()


class MockDatetime(datetime.datetime):
    """Fake Datetime"""

    @classmethod
    def utcnow(cls):
        """Return our random/fixed datetime"""
        return default_datetime()


@contextlib.contextmanager
def mocked_datetime(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    with monkeypatch.context() as m:
        m.setattr(datetime, "datetime", MockDatetime)

        yield MockDatetime


class AssertNewFile:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert os.path.exists(self.filename), self.filename
            os.remove(self.filename)
