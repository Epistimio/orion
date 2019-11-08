#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import os
import signal
import subprocess
import time

import pytest

import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.utils.format_trials import tuple_to_trial
import orion.core.worker.consumer as consumer


Consumer = consumer.Consumer


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config['metadata']['user_args'] = ['--x~uniform(-50, 50)']
    config['name'] = 'exp'
    config['working_dir'] = "/tmp/orion"
    backward.populate_space(config)
    config['space'] = config['metadata']['priors']
    return config


@pytest.mark.usefixtures("create_db_instance")
def test_trials_interrupted_keyboard_int(config, monkeypatch):
    """Check if a trial is set as interrupted when a KeyboardInterrupt is raised."""
    def mock_Popen(*args, **kwargs):
        raise KeyboardInterrupt

    exp = experiment_builder.build(**config)

    monkeypatch.setattr(consumer.subprocess, "Popen", mock_Popen)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial)

    con = Consumer(exp)

    with pytest.raises(KeyboardInterrupt):
        con.consume(trial)

    trials = exp.fetch_trials_by_status('interrupted')
    assert len(trials)
    assert trials[0].id == trial.id


@pytest.mark.usefixtures("create_db_instance")
def test_trials_interrupted_sigterm(config, monkeypatch):
    """Check if a trial is set as interrupted when a signal is raised."""
    def mock_popen(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)

    exp = experiment_builder.build(**config)

    monkeypatch.setattr(subprocess.Popen, "wait", mock_popen)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial)

    con = Consumer(exp)

    with pytest.raises(KeyboardInterrupt):
        con.consume(trial)

    trials = exp.fetch_trials_by_status('interrupted')
    assert len(trials)
    assert trials[0].id == trial.id


@pytest.mark.usefixtures("create_db_instance")
def test_pacemaker_termination(config, monkeypatch):
    """Check if pacemaker stops as soon as the trial completes."""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status='reserved')

    con = Consumer(exp)

    start = time.time()

    con.consume(trial)
    con.pacemaker.join()

    duration = time.time() - start

    assert duration < con.pacemaker.wait_time


@pytest.mark.usefixtures("create_db_instance")
def test_trial_working_dir_is_changed(config, monkeypatch):
    """Check that trial has its working_dir attribute changed."""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status='reserved')

    con = Consumer(exp)
    con.consume(trial)

    assert trial.working_dir is not None
    assert trial.working_dir == con.working_dir + "/exp_" + trial.id
