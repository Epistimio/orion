#!/usr/bin/env python
"""Perform functional tests for db dump."""

import os

import pytest

import orion.core.cli
from orion.core.io.database.pickleddb import PickledDB


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def clean_dump(dump_path):
    """Delete dumped files."""
    for path in (dump_path, f"{dump_path}.lock"):
        if os.path.isfile(path):
            os.unlink(path)


def test_dump_default(
    three_experiments_branch_same_name_trials_benchmarks, capsys, testing_helpers
):
    """Test dump with default arguments"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        testing_helpers.assert_tested_db_structure(dumped_db)
    finally:
        clean_dump("dump.pkl")


def test_dump_overwrite(
    three_experiments_branch_same_name_trials_benchmarks, capsys, testing_helpers
):
    """Test dump with overwrite argument"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        testing_helpers.assert_tested_db_structure(dumped_db)

        # No overwrite by default. Should fail.
        execute("db dump", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.strip().startswith(
            "Error: Export output already exists (specify `--force` to overwrite) at"
        )

        # Overwrite. Should pass.
        execute("db dump --force")
        assert os.path.isfile("dump.pkl")
        testing_helpers.assert_tested_db_structure(dumped_db)
    finally:
        clean_dump("dump.pkl")


def test_dump_to_specified_output(
    three_experiments_branch_same_name_trials_benchmarks, capsys, testing_helpers
):
    """Test dump to a specified output file"""
    dump_path = "test.pkl"
    assert not os.path.exists(dump_path)
    try:
        execute(f"db dump -o {dump_path}")
        assert os.path.isfile(dump_path)
        dumped_db = PickledDB(dump_path)
        testing_helpers.assert_tested_db_structure(dumped_db)
    finally:
        clean_dump(dump_path)


def test_dump_unknown_experiment(
    three_experiments_branch_same_name_trials_benchmarks, capsys
):
    """Test dump unknown experiment"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump -n i-dont-exist", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.startswith(
            "Error: No experiment found with query {'name': 'i-dont-exist'}. Nothing to dump."
        )
    finally:
        # Output file is created as soon as dst storage object is created in dump_database()
        # So, we still need to delete it here
        clean_dump("dump.pkl")


@pytest.mark.parametrize(
    "given_version,expected_version,nb_trials,nb_child_trials,algo_state",
    [
        (None, 2, 6, 0, None),
        (
            1,
            1,
            12,
            6,
            {
                "my_algo_state": "some_data",
                "my_other_state_data": "some_other_data",
            },
        ),
    ],
)
def test_dump_experiment_test_single_exp(
    three_experiments_branch_same_name_trials_benchmarks,
    testing_helpers,
    given_version,
    expected_version,
    nb_trials,
    nb_child_trials,
    algo_state,
):
    """Test dump experiment test_single_exp"""
    assert not os.path.exists("dump.pkl")
    try:
        command = "db dump -n test_single_exp"
        if given_version is not None:
            command += f" -v {given_version}"
        execute(command)
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        testing_helpers.check_unique_import(
            dumped_db,
            "test_single_exp",
            expected_version,
            nb_trials=nb_trials,
            nb_child_trials=nb_child_trials,
            algo_state=algo_state,
        )
    finally:
        clean_dump("dump.pkl")
