#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for db rm."""
import zlib

import pytest

import orion.core.cli
from orion.storage.base import get_storage


def hsh(name, version):
    return zlib.adler32(str((name, version)).encode())


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def test_no_exp(setup_pickleddb_database, capsys):
    """Test that rm non-existing exp exits gracefully"""
    execute("db rm i-dont-exist", assert_code=1)

    captured = capsys.readouterr()

    assert captured.err.startswith(
        "Error: No experiment with given name 'i-dont-exist'"
    )


def test_confirm_name(monkeypatch, single_with_trials):
    """Test name must be confirmed for deletion"""

    def incorrect_name(*args):
        return "oops"

    monkeypatch.setattr("builtins.input", incorrect_name)

    with pytest.raises(SystemExit):
        execute("db rm test_single_exp")

    def correct_name(*args):
        return "test_single_exp"

    monkeypatch.setattr("builtins.input", correct_name)

    assert len(get_storage().fetch_experiments({})) == 1
    execute("db rm test_single_exp")
    assert len(get_storage().fetch_experiments({})) == 0


def test_one_exp(single_with_trials):
    """Test that one exp is deleted properly"""
    assert len(get_storage().fetch_experiments({})) == 1
    assert len(get_storage()._fetch_trials({})) > 0
    execute("db rm -f test_single_exp")
    assert len(get_storage().fetch_experiments({})) == 0
    assert len(get_storage()._fetch_trials({})) == 0


def test_rm_all_evc(three_family_branch_with_trials):
    """Test that deleting root removes all experiments"""
    assert len(get_storage().fetch_experiments({})) == 3
    assert len(get_storage()._fetch_trials({})) > 0
    execute("db rm -f test_double_exp --version 1")
    assert len(get_storage().fetch_experiments({})) == 0
    assert len(get_storage()._fetch_trials({})) == 0


def test_rm_under_evc(three_family_branch_with_trials):
    """Test that deleting an experiment removes all children"""
    assert len(get_storage().fetch_experiments({})) == 3
    assert len(get_storage()._fetch_trials({})) > 0
    execute("db rm -f test_double_exp_child --version 1")
    assert len(get_storage().fetch_experiments({})) == 1
    assert len(get_storage()._fetch_trials({})) > 0
    # TODO: Test that the correct trials were deleted


def test_rm_default_leaf(three_experiments_same_name):
    """Test that deleting an experiment removes the leaf by default"""
    assert len(get_storage().fetch_experiments({})) == 3
    execute("db rm -f test_single_exp")
    assert len(get_storage().fetch_experiments({})) == 2


def test_rm_trials_by_status(single_with_trials):
    """Test that trials can be deleted by status"""
    trials = get_storage()._fetch_trials({})
    n_broken = sum(trial.status == "broken" for trial in trials)
    assert n_broken > 0
    execute("db rm -f test_single_exp --status broken")
    assert len(get_storage()._fetch_trials({})) == len(trials) - n_broken


def test_rm_trials_all(single_with_trials):
    """Test that trials all be deleted with '*'"""
    assert len(get_storage()._fetch_trials({})) > 0
    execute("db rm -f test_single_exp --status *")
    assert len(get_storage()._fetch_trials({})) == 0


def test_rm_trials_in_evc(three_family_branch_with_trials):
    """Test that trials of parent experiment are not deleted"""
    assert len(get_storage().fetch_experiments({})) == 3
    assert (
        len(get_storage()._fetch_trials({"experiment": hsh("test_double_exp", 1)})) > 0
    )
    assert (
        len(
            get_storage()._fetch_trials({"experiment": hsh("test_double_exp_child", 1)})
        )
        > 0
    )
    assert (
        len(
            get_storage()._fetch_trials(
                {"experiment": hsh("test_double_exp_grand_child", 1)}
            )
        )
        > 0
    )
    execute("db rm -f test_double_exp_child --status *")
    # Make sure no experiments were deleted
    assert len(get_storage().fetch_experiments({})) == 3
    # Make sure only trials of given experiment were deleted
    assert (
        len(get_storage()._fetch_trials({"experiment": hsh("test_double_exp", 1)})) > 0
    )
    assert (
        len(
            get_storage()._fetch_trials({"experiment": hsh("test_double_exp_child", 1)})
        )
        == 0
    )
    assert (
        len(
            get_storage()._fetch_trials(
                {"experiment": hsh("test_double_exp_grand_child", 1)}
            )
        )
        == 0
    )
