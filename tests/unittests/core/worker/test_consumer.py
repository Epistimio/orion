#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""

import pytest

from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils.format_trials import tuple_to_trial
import orion.core.worker.consumer as consumer


Consumer = consumer.Consumer


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config['metadata']['user_args'] = ['--x~uniform(-50, 50)']
    config['name'] = 'exp'
    return config


@pytest.mark.usefixtures("create_db_instance")
def test_trials_interrupted_keyboard_int(config, monkeypatch):
    """Check if a trial is set as interrupted when a KeyboardInterrupt is raised."""
    def mock_Popen(*args, **kwargs):
        raise KeyboardInterrupt

    exp = ExperimentBuilder().build_from(config)

    monkeypatch.setattr(consumer.subprocess, "Popen", mock_Popen)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial)

    con = Consumer(exp)

    with pytest.raises(KeyboardInterrupt):
        con.consume(trial)

    trials = exp.fetch_trials({'status': 'interrupted'})
    assert len(trials)
