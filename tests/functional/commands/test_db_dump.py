#!/usr/bin/env python
"""Perform functional tests for db dump."""

import os

import pytest

import orion.core.cli
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def clean_dump(dump_path):
    """Delete dumped files."""
    for path in (dump_path, f"{dump_path}.lock"):
        if os.path.isfile(path):
            os.unlink(path)


def test_default_storage(three_experiments_branch_same_name):
    """Check default storage from three_experiments_branch_same_name"""
    storage = setup_storage()
    experiments = storage._db.read("experiments")
    algos = storage._db.read("algo")
    assert len(experiments) == 3
    assert len(algos) == 3


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
    three_experiments_branch_same_name_trials_benchmarks,
    capsys,
    testing_helpers,
    tmp_path,
):
    """Test dump with overwrite argument"""
    dump_path = f"{tmp_path}/dump.pkl"
    try:
        execute(f"db dump -o {dump_path}")
        assert os.path.isfile(dump_path)
        dumped_db = PickledDB(dump_path)
        testing_helpers.assert_tested_db_structure(dumped_db)

        # No overwrite by default. Should fail.
        execute(f"db dump -o {dump_path}", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.strip().startswith(
            "Error: Export output already exists (specify `--force` to overwrite) at"
        )

        # Overwrite. Should pass.
        execute(f"db dump --force -o {dump_path}")
        assert os.path.isfile(dump_path)
        testing_helpers.assert_tested_db_structure(dumped_db)
    finally:
        clean_dump(dump_path)


def test_dump_to_specified_output(
    three_experiments_branch_same_name_trials_benchmarks,
    capsys,
    testing_helpers,
    tmp_path,
):
    """Test dump to a specified output file"""
    dump_path = f"{tmp_path}/test.pkl"
    assert not os.path.exists(dump_path)
    try:
        execute(f"db dump -o {dump_path}")
        assert os.path.isfile(dump_path)
        dumped_db = PickledDB(dump_path)
        testing_helpers.assert_tested_db_structure(dumped_db)
    finally:
        clean_dump(dump_path)


@pytest.mark.parametrize(
    "output_already_exists,output_specified,overwrite,error_message",
    [
        (
            True,
            False,
            False,
            "Error: Export output already exists (specify `--force` to overwrite)",
        ),
        (
            True,
            False,
            True,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
        (
            True,
            True,
            False,
            "Error: Export output already exists (specify `--force` to overwrite)",
        ),
        (
            True,
            True,
            True,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
        (
            False,
            False,
            False,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
        (
            False,
            False,
            True,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
        (
            False,
            True,
            False,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
        (
            False,
            True,
            True,
            "Error: No experiment found with query {'name': 'unknown-experiment'}. "
            "Nothing to dump.",
        ),
    ],
)
def test_dump_post_clean_on_error(
    output_already_exists, output_specified, overwrite, error_message, capsys, tmp_path
):
    """Test how dumped file is cleaned if dump fails."""

    # Prepare a command that will fail (by looking for unknown experiment)
    command = ["db", "dump", "-n", "unknown-experiment"]
    if output_specified:
        output_specified = f"{tmp_path}/test.pkl"
        command += ["--output", output_specified]
    if overwrite:
        command += ["--force"]

    expected_output = output_specified or "dump.pkl"

    # Create expected file if necessary
    output_modified_time = None
    if output_already_exists:
        assert not os.path.exists(expected_output), expected_output
        with open(expected_output, "w"):
            pass
        assert os.path.isfile(expected_output)
        output_modified_time = os.stat(expected_output).st_mtime

    # Execute command and expect it to fail
    execute(" ".join(command), assert_code=1)
    err = capsys.readouterr().err

    # Check output error
    assert err.startswith(error_message)

    # Check dump post-clean
    if output_already_exists:
        # Output should exist after error.
        assert os.path.isfile(expected_output)
        # Output should have not been modified.
        assert output_modified_time == os.stat(expected_output).st_mtime
        # Clean files anyway
        os.unlink(expected_output)
        if os.path.isfile(f"{expected_output}.lock"):
            os.unlink(f"{expected_output}.lock")
    else:
        # Output should not exist after error.
        assert not os.path.exists(expected_output)
        assert not os.path.exists(f"{expected_output}.lock")


def test_dump_unknown_experiment(
    three_experiments_branch_same_name_trials_benchmarks, capsys, tmp_path
):
    """Test dump unknown experiment"""
    dump_path = f"{tmp_path}/dump.pkl"
    try:
        execute(f"db dump -n i-dont-exist -o {dump_path}", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.startswith(
            "Error: No experiment found with query {'name': 'i-dont-exist'}. Nothing to dump."
        )
    finally:
        # Output file is created as soon as dst storage object is created in dump_database()
        # So, we still need to delete it here
        clean_dump(dump_path)


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
    tmp_path,
):
    """Test dump experiment test_single_exp"""
    dump_path = f"{tmp_path}/dump.pkl"
    try:
        command = f"db dump -n test_single_exp -o {dump_path}"
        if given_version is not None:
            command += f" -v {given_version}"
        execute(command)
        assert os.path.isfile(dump_path)
        dumped_db = PickledDB(dump_path)
        testing_helpers.check_unique_import(
            dumped_db,
            "test_single_exp",
            expected_version,
            nb_trials=nb_trials,
            nb_child_trials=nb_child_trials,
            algo_state=algo_state,
        )
    finally:
        clean_dump(dump_path)
