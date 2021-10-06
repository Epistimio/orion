#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.experiment`."""
from collections import defaultdict

import pytest

from orion.core.utils.exceptions import BrokenExperiment
from orion.testing import create_experiment

config = dict(
    name="supernaekei",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "tsirif",
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
    max_trials=10,
    max_broken=5,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
    refers=dict(root_id="supernaekei", parent_id=None, adapter=[]),
)

base_trial = {
    "experiment": 0,
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


class OrionExtensionTest:
    """Base orion extension interface you need to implement"""

    def __init__(self) -> None:
        self.calls = defaultdict(int)

    def on_experiment_error(self, *args, **kwargs):
        self.calls["on_experiment_error"] += 1

    def on_trial_error(self, *args, **kwargs):
        self.calls["on_trial_error"] += 1

    def start_experiment(self, *args, **kwargs):
        self.calls["start_experiment"] += 1

    def new_trial(self, *args, **kwargs):
        self.calls["new_trial"] += 1

    def end_trial(self, *args, **kwargs):
        self.calls["end_trial"] += 1

    def end_experiment(self, *args, **kwargs):
        self.calls["end_experiment"] += 1


def test_client_extension():
    ext = OrionExtensionTest()
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        registered_callback = client.extensions.register(ext)
        assert registered_callback == 6, "All ext callbacks got registered"

        def foo(x):
            if len(client.fetch_trials()) > 5:
                raise RuntimeError()
            return [dict(name="result", type="objective", value=x * 2)]

        MAX_TRIALS = 10
        MAX_BROKEN = 5
        assert client.max_trials == MAX_TRIALS

        with pytest.raises(BrokenExperiment):
            client.workon(foo, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)

        n_trials = len(experiment.fetch_trials_by_status("completed"))
        n_broken = len(experiment.fetch_trials_by_status("broken"))
        n_reserved = len(experiment.fetch_trials_by_status("reserved"))

        assert (
            ext.calls["new_trial"] == n_trials + n_broken - n_reserved
        ), "all trials should have triggered callbacks"
        assert (
            ext.calls["end_trial"] == n_trials + n_broken - n_reserved
        ), "all trials should have triggered callbacks"
        assert (
            ext.calls["on_trial_error"] == n_broken
        ), "failed trial should be reported "

        assert ext.calls["start_experiment"] == 1, "experiment should have started"
        assert ext.calls["end_experiment"] == 1, "experiment should have ended"
        assert ext.calls["on_experiment_error"] == 1, "failed experiment "

        unregistered_callback = client.extensions.unregister(ext)
        assert unregistered_callback == 6, "All ext callbacks got unregistered"


class BadOrionExtensionTest:
    """Base orion extension interface you need to implement"""

    def __init__(self) -> None:
        self.calls = defaultdict(int)

    def on_extension_error(self, name, fun, exception, args):
        self.calls["on_extension_error"] += 1

    def on_experiment_error(self, *args, **kwargs):
        self.calls["on_experiment_error"] += 1

    def on_trial_error(self, *args, **kwargs):
        self.calls["on_trial_error"] += 1

    def new_trial(self, *args, **kwargs):
        raise RuntimeError()


def test_client_bad_extension():
    ext = BadOrionExtensionTest()
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        registered_callback = client.extensions.register(ext)
        assert registered_callback == 4, "All ext callbacks got registered"

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        MAX_TRIALS = 10
        MAX_BROKEN = 5
        assert client.max_trials == MAX_TRIALS
        client.workon(foo, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)

        assert ext.calls["on_trial_error"] == 0, "Orion worked as expected"
        assert ext.calls["on_experiment_error"] == 0, "Orion worked as expected"
        assert ext.calls["on_extension_error"] == 9, "Extension error got reported"

        unregistered_callback = client.extensions.unregister(ext)
        assert unregistered_callback == 4, "All ext callbacks got unregistered"
