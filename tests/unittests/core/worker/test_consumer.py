#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

import orion.core.io.experiment_builder as experiment_builder
import orion.core.io.resolve_config as resolve_config
import orion.core.utils.backward as backward
import orion.core.worker.consumer as consumer
from orion.core.utils import sigterm_as_interrupt
from orion.core.utils.exceptions import BranchingEvent, MissingResultFile
from orion.core.utils.format_trials import tuple_to_trial

Consumer = consumer.Consumer


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config["metadata"]["user_args"] = ["--x~uniform(-50, 50)"]
    config["metadata"]["VCS"] = resolve_config.infer_versioning_metadata(
        config["metadata"]["user_script"]
    )
    config["name"] = "exp"
    config["working_dir"] = "/tmp/orion"
    backward.populate_space(config)
    config["space"] = config["metadata"]["priors"]
    return config


@pytest.mark.usefixtures("storage")
def test_trials_interrupted_sigterm(config, monkeypatch):
    """Check if a trial is set as interrupted when a signal is raised."""

    def mock_popen(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)

    exp = experiment_builder.build(**config)

    monkeypatch.setattr(subprocess.Popen, "wait", mock_popen)

    trial = tuple_to_trial((1.0,), exp.space)
    exp.register_trial(trial)

    con = Consumer(exp)

    with pytest.raises(KeyboardInterrupt):
        with sigterm_as_interrupt():
            con(trial)

    shutil.rmtree(trial.working_dir)


@pytest.mark.usefixtures("storage")
def test_trial_working_dir_is_created(config):
    """Check that trial working dir is created."""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status="reserved")

    assert not os.path.exists(trial.working_dir)

    con = Consumer(exp)
    con(trial)

    assert os.path.exists(trial.working_dir)

    shutil.rmtree(trial.working_dir)


def setup_code_change_mock(config, monkeypatch, ignore_code_changes):
    """Mock create experiment and trials, and infer_versioning_metadata"""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status="reserved")

    con = Consumer(exp, ignore_code_changes=ignore_code_changes)

    def code_changed(user_script):
        return dict(
            type="git",
            is_dirty=True,
            HEAD_sha="changed",
            active_branch="new_branch",
            diff_sha="new_diff",
        )

    monkeypatch.setattr(consumer, "infer_versioning_metadata", code_changed)

    return con, trial


@pytest.mark.usefixtures("storage")
def test_code_changed_evc_disabled(config, monkeypatch, caplog):
    """Check that trial has its working_dir attribute changed."""

    con, trial = setup_code_change_mock(config, monkeypatch, ignore_code_changes=True)

    with caplog.at_level(logging.WARNING):
        con(trial)
        assert "Code changed between execution of 2 trials" in caplog.text

    shutil.rmtree(trial.working_dir)


@pytest.mark.usefixtures("storage")
def test_code_changed_evc_enabled(config, monkeypatch):
    """Check that trial has its working_dir attribute changed."""

    con, trial = setup_code_change_mock(config, monkeypatch, ignore_code_changes=False)

    with pytest.raises(BranchingEvent) as exc:
        con(trial)

    assert exc.match("Code changed between execution of 2 trials")

    shutil.rmtree(trial.working_dir)


@pytest.mark.usefixtures("storage")
def test_retrieve_result_nofile(config):
    """Test retrieve result"""
    results_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="results_", suffix=".log", dir=".", delete=True
    )

    exp = experiment_builder.build(**config)

    con = Consumer(exp)

    with pytest.raises(MissingResultFile) as exec:
        con.retrieve_results(results_file)

    results_file.close()

    assert exec.match(r"Cannot parse result file")
