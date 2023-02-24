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


def _check_db(
    db: PickledDB, nb_exps, nb_algos, nb_trials, nb_benchmarks, nb_child_exps=0
):
    """Check number of expected data in given database."""
    experiments = db.read("experiments")
    assert len(experiments) == nb_exps
    assert len(db.read("algo")) == nb_algos
    assert len(db.read("trials")) == nb_trials
    assert len(db.read("benchmarks")) == nb_benchmarks

    # Check we have expected number of child experiments.
    exp_map = {exp["_id"]: exp for exp in experiments}
    assert len(exp_map) == nb_exps
    child_exps = []
    for exp in experiments:
        parent = exp["refers"]["parent_id"]
        if parent is not None:
            assert parent in exp_map
            child_exps.append(exp)
    assert len(child_exps) == nb_child_exps


def _check_exp(
    db: PickledDB, name, version, nb_trials, nb_child_trials=0, algo_state=None
):
    """Check experiment.
    - Check if we found experiment.
    - Check if we found exactly 1 algorithm for this experiment.
    - Check algo state if algo_state is provided
    - Check if we found expected number of trials for this experiment.
    - Check if we found expecter number of child trials into experiment trials.
    """
    experiments = db.read("experiments", {"name": name, "version": version})
    assert len(experiments) == 1
    (experiment,) = experiments
    algos = db.read("algo", {"experiment": experiment["_id"]})
    trials = db.read("trials", {"experiment": experiment["_id"]})
    assert len(algos) == 1
    assert len(trials) == nb_trials

    if algo_state is not None:
        (algo,) = algos
        assert algo_state == pickle.loads(algo["state"])

    trial_map = {trial["id"]: trial for trial in trials}
    assert len(trial_map) == nb_trials
    child_trials = []
    for trial in trials:
        parent = trial["parent"]
        if parent is not None:
            assert parent in trial_map
            child_trials.append(trial)
    assert len(child_trials) == nb_child_trials


def _assert_tested_db_structure(dumped_db):
    """Check counts and experiments for database from specific fixture
    `three_experiments_branch_same_name_trials_benchmarks`.
    """
    _check_db(
        dumped_db,
        nb_exps=3,
        nb_algos=3,
        nb_trials=24,
        nb_benchmarks=3,
        nb_child_exps=2,
    )
    _check_exp(dumped_db, "test_single_exp", 1, nb_trials=12, nb_child_trials=6)
    _check_exp(dumped_db, "test_single_exp", 2, nb_trials=6)
    _check_exp(dumped_db, "test_single_exp_child", 1, nb_trials=6)


def test_dump_default(three_experiments_branch_same_name_trials_benchmarks, capsys):
    """Test dump with default arguments"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        _assert_tested_db_structure(dumped_db)
    finally:
        clean_dump("dump.pkl")


def test_dump_overwrite(three_experiments_branch_same_name_trials_benchmarks, capsys):
    """Test dump with overwrite argument"""
    assert not os.path.exists("dump.pkl")
    try:
        execute("db dump")
        assert os.path.isfile("dump.pkl")
        dumped_db = PickledDB("dump.pkl")
        _assert_tested_db_structure(dumped_db)

        # No overwrite by default. Should fail.
        execute("db dump", assert_code=1)
        captured = capsys.readouterr()
        assert captured.err.strip().startswith(
            "Error: Export output already exists (specify `--force` to overwrite) at"
        )

        # Overwrite. Should pass.
        execute("db dump --force")
        assert os.path.isfile("dump.pkl")
        _assert_tested_db_structure(dumped_db)
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
        _assert_tested_db_structure(dumped_db)
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
        _check_db(dumped_db, nb_exps=1, nb_algos=1, nb_trials=6, nb_benchmarks=0)
        _check_exp(dumped_db, "test_single_exp", 2, nb_trials=6)
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
        _check_db(dumped_db, nb_exps=1, nb_algos=1, nb_trials=12, nb_benchmarks=0)
        _check_exp(
            dumped_db,
            "test_single_exp",
            1,
            nb_trials=12,
            nb_child_trials=6,
            algo_state=src_alg.state,
        )
    finally:
        clean_dump("dump.pkl")
