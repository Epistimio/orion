#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for db set."""
import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
from orion.storage.base import get_storage


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def test_no_exp(storage, capsys):
    """Test that set non-existing exp exits gracefully"""
    execute("db set i-dont-exist whatever=1 idontcare=2", assert_code=1)

    captured = capsys.readouterr()

    assert captured.err.startswith(
        "Error: No experiment with given name 'i-dont-exist'"
    )


def test_confirm_name(monkeypatch, single_with_trials):
    """Test name must be confirmed for update"""

    def incorrect_name(*args):
        return "oops"

    monkeypatch.setattr("builtins.input", incorrect_name)

    execute("db set test_single_exp status=broken status=interrupted", assert_code=1)

    def correct_name(*args):
        return "test_single_exp"

    monkeypatch.setattr("builtins.input", correct_name)

    assert len(get_storage()._fetch_trials({"status": "broken"})) > 0
    execute("db set test_single_exp status=broken status=interrupted")
    assert len(get_storage()._fetch_trials({"status": "broken"})) == 0


def test_invalid_query(single_with_trials, capsys):
    """Test error message when query is invalid"""
    execute("db set -f test_single_exp whatever=1 idontcare=2", assert_code=1)

    captured = capsys.readouterr()

    assert captured.err.startswith("Error: Invalid query attribute `whatever`.")


def test_invalid_update(single_with_trials, capsys):
    """Test error message when update attribute is invalid"""
    execute("db set -f test_single_exp status=new yoopidoo=2", assert_code=1)

    captured = capsys.readouterr()

    assert captured.err.startswith("Error: Invalid update attribute `yoopidoo`.")


def test_update_trial(single_with_trials, capsys):
    """Test that trial is updated properly"""
    trials = get_storage()._fetch_trials({})
    assert sum(trial.status == "broken" for trial in trials) > 0
    trials = dict(zip((trial.id for trial in trials), trials))
    execute("db set -f test_single_exp status=broken status=interrupted")
    for trial in get_storage()._fetch_trials({}):
        if trials[trial.id].status == "broken":
            assert trial.status == "interrupted", "status not changed properly"
        else:
            assert (
                trials[trial.id].status == trial.status
            ), "status should not have been changed"

    captured = capsys.readouterr()

    assert captured.out.endswith("1 trials modified\n")


def test_update_trial_with_id(single_with_trials, capsys):
    """Test that trial is updated properly when querying with the id"""
    trials = get_storage()._fetch_trials({})
    trials = dict(zip((trial.id for trial in trials), trials))
    trial = get_storage()._fetch_trials({"status": "broken"})[0]
    execute(f"db set -f test_single_exp id={trial.id} status=interrupted")
    for new_trial in get_storage()._fetch_trials({}):
        if new_trial.id == trial.id:
            assert new_trial.status == "interrupted", "status not changed properly"
        else:
            status_unchanged = trials[new_trial.id].status == new_trial.status
            assert status_unchanged, "status should not have been changed"

    captured = capsys.readouterr()

    assert captured.out.endswith("1 trials modified\n")


def test_update_no_match_query(single_with_trials, capsys):
    """Test that no trials are updated when there is no match"""
    trials = get_storage()._fetch_trials({})
    trials = dict(zip((trial.id for trial in trials), trials))
    execute("db set -f test_single_exp status=invalid status=interrupted")
    for trial in get_storage()._fetch_trials({}):
        assert (
            trials[trial.id].status == trial.status
        ), "status should not have been changed"

    captured = capsys.readouterr()

    assert captured.out.endswith("0 trials modified\n")


def test_update_child_only(three_experiments_same_name_with_trials, capsys):
    """Test trials of the parent experiment are not updated"""
    exp1 = experiment_builder.load(name="test_single_exp", version=1)
    exp2 = experiment_builder.load(name="test_single_exp", version=2)
    exp3 = experiment_builder.load(name="test_single_exp", version=3)
    execute("db set -f test_single_exp --version 2 status=broken status=interrupted")
    assert len(exp1.fetch_trials_by_status("broken")) > 0
    assert len(exp2.fetch_trials_by_status("broken")) == 0
    assert len(exp3.fetch_trials_by_status("broken")) == 0

    captured = capsys.readouterr()

    assert captured.out.endswith("2 trials modified\n")


def test_update_default_leaf(three_experiments_same_name_with_trials, capsys):
    """Test trials of the parent experiment are not updated"""
    exp1 = experiment_builder.load(name="test_single_exp", version=1)
    exp2 = experiment_builder.load(name="test_single_exp", version=2)
    exp3 = experiment_builder.load(name="test_single_exp", version=3)
    execute("db set -f test_single_exp status=broken status=interrupted")
    assert len(exp1.fetch_trials_by_status("broken")) > 0
    assert len(exp2.fetch_trials_by_status("broken")) > 0
    assert len(exp3.fetch_trials_by_status("broken")) == 0

    captured = capsys.readouterr()

    assert captured.out.endswith("1 trials modified\n")


def test_no_update_id_from_parent(three_experiments_same_name_with_trials, capsys):
    """Test trials of the parent experiment are not updated"""
    exp1 = experiment_builder.load(name="test_single_exp", version=1)
    exp2 = experiment_builder.load(name="test_single_exp", version=2)
    exp3 = experiment_builder.load(name="test_single_exp", version=3)
    trial = exp1.fetch_trials_by_status("broken")[0]
    execute(f"db set -f test_single_exp id={trial.id} status=shouldnothappen")
    assert len(exp1.fetch_trials_by_status("shouldnothappen")) == 0
    assert len(exp2.fetch_trials_by_status("shouldnothappen")) == 0
    assert len(exp3.fetch_trials_by_status("shouldnothappen")) == 0

    captured = capsys.readouterr()

    assert captured.out.endswith("0 trials modified\n")
