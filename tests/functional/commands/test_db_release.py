#!/usr/bin/env python
"""Perform functional tests for db release."""

import pytest

import orion.core.cli
from orion.storage.base import setup_storage


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def test_no_exp(orionstate, capsys):
    """Test that releasing non-existing exp exits gracefully"""
    execute("db release i-dont-exist", assert_code=1)

    captured = capsys.readouterr()

    assert captured.err.startswith(
        "Error: No experiment with given name 'i-dont-exist'"
    )


def test_confirm_name(monkeypatch, single_with_trials):
    """Test name must be confirmed for release"""

    def incorrect_name(*args):
        return "oops"

    monkeypatch.setattr("builtins.input", incorrect_name)

    with pytest.raises(SystemExit):
        execute("db release test_single_exp")

    def correct_name(*args):
        return "test_single_exp"

    monkeypatch.setattr("builtins.input", correct_name)

    experiments = setup_storage().fetch_experiments({})
    uid = experiments[0]["_id"]
    with setup_storage().acquire_algorithm_lock(uid=uid) as algo_state_lock:
        assert algo_state_lock.state is None
        algo_state_lock.set_state({})

    with setup_storage().acquire_algorithm_lock(uid=uid) as algo_state_lock:
        assert algo_state_lock.state == {}
        assert setup_storage().get_algorithm_lock_info(uid=uid).locked == 1
        execute("db release test_single_exp")
        assert setup_storage().get_algorithm_lock_info(uid=uid).locked == 0
        assert setup_storage().get_algorithm_lock_info(uid=uid).state == {}


def test_one_exp(single_with_trials):
    """Test that one exp is deleted properly"""
    experiments = setup_storage().fetch_experiments({})
    uid = experiments[0]["_id"]
    assert setup_storage().get_algorithm_lock_info(uid=uid).locked == 0
    with setup_storage().acquire_algorithm_lock(uid=uid):
        assert setup_storage().get_algorithm_lock_info(uid=uid).locked == 1
        execute("db release -f test_single_exp")
        assert setup_storage().get_algorithm_lock_info(uid=uid).locked == 0


def test_release_name(three_family_branch_with_trials):
    """Test that deleting an experiment removes all children"""
    experiments = setup_storage().fetch_experiments({})
    storage = setup_storage()
    assert len(experiments) == 3
    assert len(storage._fetch_trials({})) > 0
    uid = None
    for experiment in experiments:
        if experiment["name"] == "test_double_exp_child":
            uid = experiment["_id"]
        assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
    assert uid is not None

    with storage.acquire_algorithm_lock(uid=uid):
        assert storage.get_algorithm_lock_info(uid=uid).locked == 1
        for experiment in experiments:
            if experiment["name"] == "test_double_exp_child":
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 1
                )
            else:
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
                )

        execute("db release -f test_double_exp_child")
        for experiment in experiments:
            assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0


def test_release_version(three_experiments_same_name_with_trials):
    """Test releasing a specific experiment version"""
    experiments = setup_storage().fetch_experiments({})
    storage = setup_storage()
    assert len(experiments) == 3
    assert len(storage._fetch_trials({})) > 0
    uid = None
    for experiment in experiments:
        if experiment["version"] == 2:
            uid = experiment["_id"]
        assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
    assert uid is not None

    with storage.acquire_algorithm_lock(uid=uid):
        assert storage.get_algorithm_lock_info(uid=uid).locked == 1
        for experiment in experiments:
            if experiment["version"] == 2:
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 1
                )
            else:
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
                )

        execute("db release -f test_single_exp --version 2")
        for experiment in experiments:
            assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0


def test_release_default_leaf(three_experiments_same_name_with_trials):
    """Test that release an experiment releases the leaf by default"""
    experiments = setup_storage().fetch_experiments({})
    storage = setup_storage()
    assert len(experiments) == 3
    assert len(storage._fetch_trials({})) > 0
    uid = None
    for experiment in experiments:
        if experiment["version"] == 3:
            uid = experiment["_id"]
        assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
    assert uid is not None

    with storage.acquire_algorithm_lock(uid=uid):
        assert storage.get_algorithm_lock_info(uid=uid).locked == 1
        for experiment in experiments:
            if experiment["version"] == 3:
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 1
                )
            else:
                assert (
                    storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
                )

        execute("db release -f test_single_exp")
        for experiment in experiments:
            assert storage.get_algorithm_lock_info(uid=experiment["_id"]).locked == 0
