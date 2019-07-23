#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import datetime
import time

import pytest

from orion.core.io.database import Database
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.trial_pacemaker import TrialPacemaker


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config['metadata']['user_args'] = ['--x~uniform(-50, 50)']
    config['name'] = 'exp'
    return config


@pytest.fixture
def exp(config):
    """Return an Experiment."""
    return ExperimentBuilder().build_from(config)


@pytest.fixture
def trial(exp):
    """Return a Trial which is registered in DB."""
    trial = tuple_to_trial((1.0,), exp.space)
    heartbeat = datetime.datetime.utcnow()
    trial.experiment = exp.id
    trial.status = 'reserved'
    trial.heartbeat = heartbeat

    Database().write('trials', trial.to_dict())

    return trial


@pytest.mark.usefixtures("create_db_instance")
def test_trial_update_heartbeat(exp, trial):
    """Test that the heartbeat of a trial has been updated."""
    trial_monitor = TrialPacemaker(exp, trial.id, wait_time=1)

    trial_monitor.start()
    time.sleep(2)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})

    assert trial.heartbeat != trials[0].heartbeat

    heartbeat = trials[0].heartbeat

    time.sleep(2)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})

    assert heartbeat != trials[0].heartbeat
    trial_monitor.stop()


@pytest.mark.usefixtures("create_db_instance")
def test_trial_heartbeat_not_updated(exp, trial):
    """Test that the heartbeat of a trial is not updated when trial is not longer reserved."""
    trial_monitor = TrialPacemaker(exp, trial.id, wait_time=1)

    trial_monitor.start()
    time.sleep(2)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})

    assert trial.heartbeat != trials[0].heartbeat

    data = {'status': 'interrupted'}
    Database().write('trials', data, query=dict(_id=trial.id))

    time.sleep(2)

    # `join` blocks until all thread have finish executing. So, the test will hang if it fails.
    trial_monitor.join()
    assert 1


@pytest.mark.usefixtures("create_db_instance")
def test_trial_heartbeat_not_updated_inbetween(exp, trial):
    """Test that the heartbeat of a trial is not updated before wait time."""
    trial_monitor = TrialPacemaker(exp, trial.id, wait_time=5)

    trial_monitor.start()
    time.sleep(1)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})
    assert trial.heartbeat.replace(microsecond=0) == trials[0].heartbeat.replace(microsecond=0)

    heartbeat = trials[0].heartbeat

    time.sleep(6)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})

    assert heartbeat != trials[0].heartbeat
    trial_monitor.stop()
