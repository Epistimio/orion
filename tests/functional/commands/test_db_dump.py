#!/usr/bin/env python
"""Perform functional tests for db dump."""

import os
import pickle

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


def test_dump_default(three_experiments_branch_same_name_trials_benchmarks, capsys):
    """Test dump with default arguments"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        with dumped_db.locked_database(write=False) as internal_db:
            collections = set(internal_db._db.keys())
        assert collections == {"experiments", "algo", "trials", "benchmarks"}
        assert len(dumped_db.read("experiments")) == 3
        assert len(dumped_db.read("algo")) == 3
        assert len(dumped_db.read("trials")) == 24
        assert len(dumped_db.read("benchmarks")) == 3
    finally:
        clean_dump("dump.pkl")


def test_dump_overwrite(three_experiments_branch_same_name_trials_benchmarks, capsys):
    """Test dump with overwrite argument"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        assert len(dumped_db.read("experiments")) == 3
        assert len(dumped_db.read("algo")) == 3
        assert len(dumped_db.read("trials")) == 24
        assert len(dumped_db.read("benchmarks")) == 3

        # No overwrite by default. Should fail.
        execute("db dump", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.strip().startswith(
            "Error: Export output already exists (specify `--force` to overwrite) at"
        )

        # Overwrite. Should pass.
        execute("db dump --force")
        assert os.path.isfile("dump.pkl")
        assert len(dumped_db.read("experiments")) == 3
        assert len(dumped_db.read("algo")) == 3
        assert len(dumped_db.read("trials")) == 24
        assert len(dumped_db.read("benchmarks")) == 3
    finally:
        clean_dump("dump.pkl")


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


def test_dump_to_specified_output(
    three_experiments_branch_same_name_trials_benchmarks, capsys
):
    """Test dump to a specified output file"""
    dump_path = "test.pkl"
    assert not os.path.exists(dump_path)
    try:
        execute(f"db dump -o {dump_path}")
        assert os.path.isfile(dump_path)
        dumped_db = PickledDB(dump_path)
        with dumped_db.locked_database(write=False) as internal_db:
            collections = set(internal_db._db.keys())
        assert collections == {"experiments", "algo", "trials", "benchmarks"}
        assert len(dumped_db.read("experiments")) == 3
        assert len(dumped_db.read("algo")) == 3
        assert len(dumped_db.read("trials")) == 24
        assert len(dumped_db.read("benchmarks")) == 3
    finally:
        clean_dump(dump_path)


def test_dump_one_experiment(
    three_experiments_branch_same_name_trials_benchmarks, capsys
):
    """Test dump only experiment test_single_exp (no version specified)"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump -n test_single_exp")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        assert len(dumped_db.read("benchmarks")) == 0
        experiments = dumped_db.read("experiments")
        algos = dumped_db.read("algo")
        trials = dumped_db.read("trials")
        assert len(experiments) == 1
        (exp_data,) = experiments
        # We must have dumped version 2
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 2
        assert len(algos) == len(exp_data["algorithm"]) == 1
        # This experiment must have only 6 trials
        assert len(trials) == 6
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)
    finally:
        clean_dump("dump.pkl")


def test_dump_one_experiment_other_version(
    three_experiments_branch_same_name_trials_benchmarks, capsys
):
    """Test dump version 1 of experiment test_single_exp"""

    # Check src algo state
    src_storage = setup_storage()
    src_exps = src_storage.fetch_experiments({"name": "test_single_exp", "version": 1})
    assert len(src_exps) == 1
    (src_exp,) = src_exps
    src_alg = src_storage.get_algorithm_lock_info(uid=src_exp["_id"])
    assert src_alg.state == {
        "my_algo_state": "some_data",
        "my_other_state_data": "some_other_data",
    }

    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump -n test_single_exp -v 1")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        assert len(dumped_db.read("benchmarks")) == 0
        experiments = dumped_db.read("experiments")
        algos = dumped_db.read("algo")
        trials = dumped_db.read("trials")
        assert len(experiments) == 1
        (exp_data,) = experiments
        assert exp_data["name"] == "test_single_exp"
        assert exp_data["version"] == 1
        # Check dumped algo
        assert len(algos) == len(exp_data["algorithm"]) == 1
        (algo,) = algos
        assert src_alg.state == pickle.loads(algo["state"])
        # This experiment must have 12 trials (children included)
        assert len(trials) == 12
        assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
        assert all(trial["experiment"] == exp_data["_id"] for trial in trials)
    finally:
        clean_dump("dump.pkl")
