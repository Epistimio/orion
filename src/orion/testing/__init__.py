# -*- coding: utf-8 -*-
"""
:mod:`orion.testing` -- Common testing support module
=====================================================
.. module:: testing
   :platform: Unix
   :synopsis: Common testing support module providing defaults, functions and mocks.
"""
# pylint: disable=protected-access

import copy
import datetime
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
