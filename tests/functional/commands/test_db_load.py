#!/usr/bin/env python
"""Perform functional tests for db load."""

import os

import pytest

import orion.core.cli
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def test_empty_database(empty_database, testing_helpers):
    """Test destination database is empty as expected"""
    storage = setup_storage()
    db = storage._db
    with db.locked_database(write=False) as internal_db:
        collections = set(internal_db._db.keys())
    assert collections == {"experiments", "algo", "trials", "benchmarks"}
    testing_helpers.check_db(db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0)


def test_load_all(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load all database"""
    assert os.path.isfile(pkl_experiments_and_benchmarks)
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db)


def test_load_again_without_resolve(
    other_empty_database, pkl_experiments_and_benchmarks, capsys, testing_helpers
):
    """Test load all database twice without resolve in second call"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_db(
        loaded_db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0
    )

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db)

    # Again
    execute(f"db load {pkl_experiments_and_benchmarks}", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Conflict detected without strategy to resolve (None) for benchmark branin_baselines_webapi"
    )
    # Destination should have not changed
    testing_helpers.assert_tested_db_structure(loaded_db)


def test_load_ignore(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load all database with --resolve ignore"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db)
    testing_helpers.assert_tested_trial_status(loaded_db)

    # Change something in PKL file to check that changes are ignored
    src_db = PickledDB(pkl_experiments_and_benchmarks)
    testing_helpers.assert_tested_trial_status(src_db)
    for trial in src_db.read("trials"):
        trial["status"] = "new"
        src_db.write("trials", trial, query={"_id": trial["_id"]})
    testing_helpers.assert_tested_db_structure(src_db)
    # Trials status checking should fail for PKL file
    with pytest.raises(AssertionError):
        testing_helpers.assert_tested_trial_status(src_db)
    # ... And pass for a specific count
    testing_helpers.assert_tested_trial_status(src_db, counts={"new": 24})

    execute(f"db load {pkl_experiments_and_benchmarks} -r ignore -c {cfg_path}")
    # Duplicated data should be ignored, so we must expect same data.
    testing_helpers.assert_tested_db_structure(loaded_db)
    # Trials status should have not been modified in dst database.
    testing_helpers.assert_tested_trial_status(loaded_db)


def test_load_overwrite(
    other_empty_database, pkl_experiments_and_benchmarks, capsys, testing_helpers
):
    """Test load all database with --resolve overwrite"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(f"db load {pkl_experiments_and_benchmarks} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db)
    testing_helpers.assert_tested_trial_status(loaded_db)

    # Change something in PKL file to check that changes are ignored
    src_db = PickledDB(pkl_experiments_and_benchmarks)
    testing_helpers.assert_tested_trial_status(src_db)
    for trial in src_db.read("trials"):
        trial["status"] = "new"
        src_db.write("trials", trial, query={"_id": trial["_id"]})
    testing_helpers.assert_tested_db_structure(src_db)
    # Trials status checking should fail for PKL file
    with pytest.raises(AssertionError):
        testing_helpers.assert_tested_trial_status(src_db)
    # ... And pass for a specific count
    testing_helpers.assert_tested_trial_status(src_db, counts={"new": 24})

    execute(f"db load {pkl_experiments_and_benchmarks} -r overwrite -c {cfg_path}")
    # We expect same data structure after overwriting
    testing_helpers.assert_tested_db_structure(loaded_db)
    # Trial status checking must fail by default
    with pytest.raises(AssertionError):
        testing_helpers.assert_tested_trial_status(loaded_db)
    # ... And pass for specific changes
    testing_helpers.assert_tested_trial_status(loaded_db, counts={"new": 24})

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


def test_load_bump_no_benchmarks(
    other_empty_database, pkl_experiments, testing_helpers
):
    """Test load all database with --resolve --bump"""
    data_source = pkl_experiments

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(f"db load {data_source} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db, nb_orig_benchmarks=0)

    execute(f"db load {data_source} -r bump -c {cfg_path}")
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    testing_helpers.assert_tested_db_structure(
        loaded_db, nb_orig_benchmarks=0, nb_duplicated=2
    )

    execute(f"db load {data_source} -r bump -c {cfg_path}")
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    testing_helpers.assert_tested_db_structure(
        loaded_db, nb_orig_benchmarks=0, nb_duplicated=3
    )


def test_load_bump_with_benchmarks(
    other_empty_database, pkl_experiments_and_benchmarks, capsys, testing_helpers
):
    """Test load all database with benchmarks and --resolve --bump"""
    data_source = pkl_experiments_and_benchmarks

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    # First execution should pass, as destination contains nothing.
    execute(f"db load {data_source} -c {cfg_path}")
    testing_helpers.assert_tested_db_structure(loaded_db)

    # New execution should fail, as benchmarks don't currently support bump.
    execute(f"db load {data_source} -r bump", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Can't bump benchmark version, as benchmarks do not currently support versioning."
    )
    # Destination should have not changed.
    testing_helpers.assert_tested_db_structure(loaded_db)


def test_load_one_experiment(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load experiment test_single_exp"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -c {cfg_path}"
    )
    testing_helpers.check_db(
        loaded_db, nb_exps=1, nb_algos=1, nb_trials=6, nb_benchmarks=0
    )
    testing_helpers.check_exp(loaded_db, "test_single_exp", 2, nb_trials=6)


def test_load_one_experiment_other_version(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load version 1 of experiment test_single_exp"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_ignore(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load experiment test_single_exp with --resolve ignore"""

    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r ignore -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_overwrite(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load experiment test_single_exp with --resolve overwrite"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r overwrite -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)


def test_load_one_experiment_bump(
    other_empty_database, pkl_experiments_and_benchmarks, testing_helpers
):
    """Test load experiment test_single_exp with --resolve bump"""
    storage, cfg_path = other_empty_database
    loaded_db = storage._db
    testing_helpers.check_empty_db(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -n test_single_exp -v 1 -c {cfg_path}"
    )
    testing_helpers.check_unique_import_test_single_expV1(loaded_db)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r bump -n test_single_exp -v 1 -c {cfg_path}"
    )
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    testing_helpers.check_unique_import_test_single_expV1(loaded_db, nb_versions=2)

    execute(
        f"db load {pkl_experiments_and_benchmarks} -r bump -n test_single_exp -v 1 -c {cfg_path}"
    )
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    testing_helpers.check_unique_import_test_single_expV1(loaded_db, nb_versions=3)
