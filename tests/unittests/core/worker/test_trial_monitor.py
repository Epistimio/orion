#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import datetime
import time

import pytest

from orion.core.io.database import Database
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils.format_trials import tuple_to_trial
from orion.core.worker.trial_monitor import TrialMonitor


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config['metadata']['user_args'] = ['--x~uniform(-50, 50)']
    config['name'] = 'exp'
    return config


@pytest.mark.usefixtures("create_db_instance")
def test_trial_update_heartbeat(config):
    """Test that the heartbeat of a trial has been updated."""
    exp = ExperimentBuilder().build_from(config)
    trial = tuple_to_trial((1.0,), exp.space)
    heartbeat = datetime.datetime.utcnow()
    trial.heartbeat = heartbeat

    data = {'_id': trial.id, 'status': 'reserved', 'heartbeat': heartbeat, 'experiment': exp.id}

    Database().write('trials', data)

    trial_monitor = TrialMonitor(exp, trial.id, wait_time=1)

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
def test_trial_heartbeat_not_updated(config):
    """Test that the heartbeat of a trial is not updated when trial is not longer reserved."""
    exp = ExperimentBuilder().build_from(config)
    trial = tuple_to_trial((1.0,), exp.space)
    heartbeat = datetime.datetime.utcnow()
    trial.heartbeat = heartbeat

    data = {'_id': trial.id, 'status': 'reserved', 'heartbeat': heartbeat, 'experiment': exp.id}

    Database().write('trials', data)

    trial_monitor = TrialMonitor(exp, trial.id, wait_time=1)

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
def test_trial_heartbeat_not_updated_inbetween(config):
    """Test that the heartbeat of a trial is not updated before wait time."""
    exp = ExperimentBuilder().build_from(config)
    trial = tuple_to_trial((1.0,), exp.space)
    heartbeat = datetime.datetime.utcnow().replace(microsecond=0)
    trial.heartbeat = heartbeat

    data = {'_id': trial.id, 'status': 'reserved', 'heartbeat': heartbeat, 'experiment': exp.id}

    Database().write('trials', data)

    trial_monitor = TrialMonitor(exp, trial.id, wait_time=5)

    trial_monitor.start()
    time.sleep(1)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})
    assert trial.heartbeat == trials[0].heartbeat

    heartbeat = trials[0].heartbeat

    time.sleep(4)

    trials = exp.fetch_trials({'_id': trial.id, 'status': 'reserved'})

    assert heartbeat != trials[0].heartbeat
    trial_monitor.stop()
