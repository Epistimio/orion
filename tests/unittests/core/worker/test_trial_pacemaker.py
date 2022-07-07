#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import datetime
import time

import pytest

import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.trial_pacemaker import TrialPacemaker
from orion.storage.base import setup_storage


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config["space"] = {"x": "uniform(-50, 50)"}
    config["name"] = "exp"
    return config


@pytest.fixture
def exp(storage, config):
    """Return an Experiment."""
    return experiment_builder.build(**config, storage=storage)


@pytest.fixture
def trial(exp):
    """Return a Trial which is registered in DB."""
    trial = tuple_to_trial((1.0,), exp.space)
    heartbeat = datetime.datetime.utcnow()
    trial.experiment = exp.id
    trial.status = "reserved"
    trial.heartbeat = heartbeat

    setup_storage().register_trial(trial)

    return trial


def test_trial_update_heartbeat(exp, trial):
    """Test that the heartbeat of a trial has been updated."""

    storage = setup_storage()

    trial_monitor = TrialPacemaker(trial, wait_time=1, storage=storage)

    trial_monitor.start()
    time.sleep(2)

    trials = exp.fetch_trials_by_status("reserved")

    assert trial.heartbeat != trials[0].heartbeat

    heartbeat = trials[0].heartbeat

    time.sleep(2)

    trials = exp.fetch_trials_by_status(status="reserved")

    assert heartbeat != trials[0].heartbeat
    trial_monitor.stop()


def test_trial_heartbeat_not_updated(exp, trial):
    """Test that the heartbeat of a trial is not updated when trial is not longer reserved."""
    storage = setup_storage()

    trial_monitor = TrialPacemaker(trial, wait_time=1, storage=storage)

    trial_monitor.start()
    time.sleep(2)

    trials = exp.fetch_trials_by_status("reserved")

    assert trial.heartbeat != trials[0].heartbeat

    setup_storage().set_trial_status(trial, status="interrupted")

    time.sleep(2)

    # `join` blocks until all thread have finish executing. So, the test will hang if it fails.
    trial_monitor.join()
    assert 1


def test_trial_heartbeat_not_updated_inbetween(exp, trial):
    """Test that the heartbeat of a trial is not updated before wait time."""
    storage = setup_storage()
    trial_monitor = TrialPacemaker(trial, wait_time=5, storage=storage)

    trial_monitor.start()
    time.sleep(1)

    trials = exp.fetch_trials_by_status("reserved")
    assert trial.heartbeat.replace(microsecond=0) == trials[0].heartbeat.replace(
        microsecond=0
    )

    heartbeat = trials[0].heartbeat

    time.sleep(6)

    trials = exp.fetch_trials_by_status(status="reserved")

    assert heartbeat != trials[0].heartbeat
    trial_monitor.stop()
