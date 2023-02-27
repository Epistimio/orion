#!/usr/bin/env python
"""Perform functional tests for db load."""

import os

import orion.core.cli
from orion.storage.base import setup_storage

from .test_db_dump import _assert_tested_db_structure, _check_db, _check_exp

# TODO: Test trial parents links and experiment root links in loaded data


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def _check_empty_db(loaded_db):
    """Check that given database is empty"""
    _check_db(loaded_db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0)


def _check_unique_import_test_single_expV1(loaded_db, nb_versions=1):
    """Check all versions of original experiment test_single_exp.1 in given database"""
    _check_db(
        loaded_db,
        nb_exps=1 * nb_versions,
        nb_algos=1 * nb_versions,
        nb_trials=12 * nb_versions,
        nb_benchmarks=0,
    )
    for i in range(nb_versions):
        _check_exp(
            loaded_db,
            "test_single_exp",
            1 + i,
            nb_trials=12,
            nb_child_trials=6,
            algo_state={
                "my_algo_state": "some_data",
                "my_other_state_data": "some_other_data",
            },
        )


def test_empty_database(empty_database):
    """Test destination database is empty as expected"""
    storage = setup_storage()
    db = storage._db
    with db.locked_database(write=False) as internal_db:
        collections = set(internal_db._db.keys())
    assert collections == {"experiments", "algo", "trials", "benchmarks"}
    _check_db(db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0)


def test_load_all(other_empty_database, pkl_experiments_and_benchmarks):
    """Test load all database"""
    assert os.path.isfile(pkl_experiments_and_benchmarks)
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    _assert_tested_db_structure(loaded_db)


def test_load_again_without_resolve(
    other_empty_database, pkl_experiments_and_benchmarks, capsys
):
    """Test load all database twice without resolve in second call"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_db(loaded_db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    _assert_tested_db_structure(loaded_db)

    # Again
    execute(f"db load {pkl_experiments_and_benchmarks}", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Conflict detected without strategy to resolve (None) for benchmark branin_baselines_webapi"
    )
    # Destination should have not changed
    _assert_tested_db_structure(loaded_db)


def test_load_ignore(other_empty_database, pkl_experiments_and_benchmarks):
    """Test load all database with --resolve ignore"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    _assert_tested_db_structure(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -r ignore -c {cfg_path}")
    # Duplicated data should be ignored, so we must expect same number of data and same IDs.
    _assert_tested_db_structure(loaded_db)


def test_load_overwrite(other_empty_database, pkl_experiments_and_benchmarks, capsys):
    """Test load all database with --resolve overwrite"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    _assert_tested_db_structure(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -r overwrite -c {cfg_path}")
    # We expect same data structure after overwriting
    _assert_tested_db_structure(loaded_db)

    # Check output to verify progress callback messages
    captured = capsys.readouterr()
    assert captured.err.strip() == ""
    assert (
        captured.out.strip()
        == """
STEP 1 Collect source experiments to load 0 1
STEP 1 Collect source experiments to load 1 1
STEP 2 Check benchmarks 0 3
STEP 2 Check benchmarks 1 3
STEP 2 Check benchmarks 2 3
STEP 2 Check benchmarks 3 3
STEP 3 Check destination experiments 0 1
STEP 3 Check destination experiments 1 1
STEP 4 Check source experiments 0 3
STEP 4 Check source experiments 1 3
STEP 4 Check source experiments 2 3
STEP 4 Check source experiments 3 3
STEP 5 Delete data to replace in destination 0 0
STEP 6 Insert new data in destination 0 6
STEP 6 Insert new data in destination 1 6
STEP 6 Insert new data in destination 2 6
STEP 6 Insert new data in destination 3 6
STEP 6 Insert new data in destination 4 6
STEP 6 Insert new data in destination 5 6
STEP 6 Insert new data in destination 6 6
STEP 1 Collect source experiments to load 0 1
STEP 1 Collect source experiments to load 1 1
STEP 2 Check benchmarks 0 3
STEP 2 Check benchmarks 1 3
STEP 2 Check benchmarks 2 3
STEP 2 Check benchmarks 3 3
STEP 3 Check destination experiments 0 1
STEP 3 Check destination experiments 1 1
STEP 4 Check source experiments 0 3
STEP 4 Check source experiments 1 3
STEP 4 Check source experiments 2 3
STEP 4 Check source experiments 3 3
STEP 5 Delete data to replace in destination 0 12
STEP 5 Delete data to replace in destination 1 12
STEP 5 Delete data to replace in destination 2 12
STEP 5 Delete data to replace in destination 3 12
STEP 5 Delete data to replace in destination 4 12
STEP 5 Delete data to replace in destination 5 12
STEP 5 Delete data to replace in destination 6 12
STEP 5 Delete data to replace in destination 7 12
STEP 5 Delete data to replace in destination 8 12
STEP 5 Delete data to replace in destination 9 12
STEP 5 Delete data to replace in destination 10 12
STEP 5 Delete data to replace in destination 11 12
STEP 5 Delete data to replace in destination 12 12
STEP 6 Insert new data in destination 0 6
STEP 6 Insert new data in destination 1 6
STEP 6 Insert new data in destination 2 6
STEP 6 Insert new data in destination 3 6
STEP 6 Insert new data in destination 4 6
STEP 6 Insert new data in destination 5 6
STEP 6 Insert new data in destination 6 6
""".strip()
    )


def test_load_bump_no_benchmarks(other_empty_database, pkl_experiments):
    """Test load all database with --resolve --bump"""
    data_source = pkl_experiments

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(f"db load {data_source} -c {cfg_path}")
    _check_db(
        loaded_db,
        nb_exps=3,
        nb_algos=3,
        nb_trials=24,
        nb_benchmarks=0,
        nb_child_exps=2,
    )
    _check_exp(loaded_db, "test_single_exp", 1, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 2, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 1, nb_trials=6)

    execute(f"db load {data_source} -r bump -c {cfg_path}")
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    _check_db(
        loaded_db,
        nb_exps=3 * 2,
        nb_algos=3 * 2,
        nb_trials=24 * 2,
        nb_benchmarks=0,
        nb_child_exps=2 * 2,
    )
    _check_exp(loaded_db, "test_single_exp", 1, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 2, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 1, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp", 3, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 4, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 2, nb_trials=6)

    execute(f"db load {data_source} -r bump -c {cfg_path}")
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    _check_db(
        loaded_db,
        nb_exps=3 * 3,
        nb_algos=3 * 3,
        nb_trials=24 * 3,
        nb_benchmarks=0,
        nb_child_exps=2 * 3,
    )
    _check_exp(loaded_db, "test_single_exp", 1, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 2, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 1, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp", 3, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 4, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 2, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp", 5, nb_trials=12, nb_child_trials=6)
    _check_exp(loaded_db, "test_single_exp", 6, nb_trials=6)
    _check_exp(loaded_db, "test_single_exp_child", 3, nb_trials=6)


def test_load_bump_with_benchmarks(
    other_empty_database, pkl_experiments_and_benchmarks, capsys
):
    """Test load all database with benchmarks and --resolve --bump"""
    data_source = pkl_experiments_and_benchmarks

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    # First execution should pass, as destination contains nothing.
    execute(f"db load {data_source} -c {cfg_path}")
    _assert_tested_db_structure(loaded_db)

    # New execution should fail, as benchmarks don't currently support bump.
    execute(f"db load {data_source} -r bump", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Can't bump benchmark version, as benchmarks do not currently support versioning."
    )
    # Destination should have not changed.
    _assert_tested_db_structure(loaded_db)


def test_load_one_experiment(other_empty_database, pkl_experiments_and_benchmarks):
    """Test load experiment test_single_exp"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -c {cfg_path}"
    )
    _check_db(loaded_db, nb_exps=1, nb_algos=1, nb_trials=6, nb_benchmarks=0)
    _check_exp(loaded_db, "test_single_exp", 2, nb_trials=6)


def test_load_one_experiment_other_version(
    other_empty_database, pkl_experiments_and_benchmarks
):
    """Test load version 1 of experiment test_single_exp"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_ignore(
    other_empty_database, pkl_experiments_and_benchmarks
):
    """Test load experiment test_single_exp with --resolve ignore"""

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r ignore -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_overwrite(
    other_empty_database, pkl_experiments_and_benchmarks
):
    """Test load experiment test_single_exp with --resolve overwrite"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r overwrite -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_bump(other_empty_database, pkl_experiments_and_benchmarks):
    """Test load experiment test_single_exp with --resolve bump"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    _check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    _check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r bump -n test_single_exp -v 1 -c {cfg_path}"
    )
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    _check_unique_import_test_single_expV1(loaded_db, nb_versions=2)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r bump -n test_single_exp -v 1 -c {cfg_path}"
    )
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    _check_unique_import_test_single_expV1(loaded_db, nb_versions=3)
