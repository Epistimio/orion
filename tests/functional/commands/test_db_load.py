#!/usr/bin/env python
"""Perform functional tests for db load."""

import os

import orion.core.cli
from orion.storage.base import setup_storage

LOAD_DATA = os.path.join(os.path.dirname(__file__), "orion_db_load_test_data.pickled")
LOAD_DATA_WITH_BENCHMARKS = os.path.join(
    os.path.dirname(__file__), "orion_db_load_test_data_with_benchmarks.pickled"
)


# TODO: Test loading benchmarks


def execute(command, assert_code=0):
    """Execute orion command and return returncode"""
    returncode = orion.core.cli.main(command.split(" "))
    assert returncode == assert_code


def common_indices(data_list1, data_list2):
    """Return set of common indices from two lists of data"""
    return {element["_id"] for element in data_list1} & {
        element["_id"] for element in data_list2
    }


def test_empty_database(empty_database):
    """Test destination database is empty as expected"""
    storage = setup_storage()
    db = storage._db
    with db.locked_database(write=False) as internal_db:
        collections = set(internal_db._db.keys())
    assert collections == {"experiments", "algo", "trials", "benchmarks"}
    assert len(db.read("experiments")) == 0
    assert len(db.read("algo")) == 0
    assert len(db.read("trials")) == 0
    assert len(db.read("benchmarks")) == 0


def test_load_all(empty_database):
    """Test load all database"""
    assert os.path.isfile(LOAD_DATA_WITH_BENCHMARKS)
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS}")
    with loaded_db.locked_database(write=False) as internal_db:
        collections = set(internal_db._db.keys())
    assert collections == {"experiments", "algo", "trials", "benchmarks"}

    assert len(loaded_db.read("benchmarks")) == 3
    assert len(loaded_db.read("experiments")) == 3
    assert len(loaded_db.read("trials")) == 24
    # TODO: We should expect 6 algorithms, but only 3 are returned
    # It seems config `three_experiments_family_same_name` contains 3 supplementary algorithms
    # that are not related to experiments registered in the database. So, when dumping from this config
    # then loading from dumped data, only algorithms related to available experiments are loaded,
    # and there are only 3 such algorithms (1 per experiment)
    assert len(loaded_db.read("algo")) == 3


def test_load_again_without_resolve(empty_database, capsys):
    """Test load all database twice without resolve in second call"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS}")
    benchmarks = loaded_db.read("benchmarks")
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(benchmarks) == 3
    assert len(experiments) == 3
    assert len(trials) == 24
    assert len(algos) == 3

    # Again
    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS}", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Conflict detected without strategy to resolve (None) for benchmark branin_baselines_webapi"
    )
    # Destination should have not changed
    new_benchmarks = loaded_db.read("benchmarks")
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert new_benchmarks == benchmarks
    assert new_experiments == experiments
    assert new_trials == trials
    assert new_algos == algos


def test_load_ignore(empty_database):
    """Test load all database with --resolve ignore"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS}")
    benchmarks = loaded_db.read("benchmarks")
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(benchmarks) == 3
    assert len(experiments) == 3
    assert len(trials) == 24
    assert len(algos) == 3

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r ignore")
    # Duplicated data should be ignored, so we must expect same number of data and same IDs.
    new_benchmarks = loaded_db.read("benchmarks")
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert len(new_benchmarks) == 3
    assert len(new_experiments) == 3
    assert len(new_trials) == 24
    assert len(new_algos) == 3
    assert len(common_indices(experiments, new_experiments)) == 3
    assert len(common_indices(trials, new_trials)) == 24
    assert len(common_indices(algos, new_algos)) == 3


def test_load_overwrite(empty_database, capsys):
    """Test load all database with --resolve overwrite"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS}")
    benchmarks = loaded_db.read("benchmarks")
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(benchmarks) == 3
    assert len(experiments) == 3
    assert len(trials) == 24
    assert len(algos) == 3

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r overwrite")
    # Duplicated data should be overwritten, so we must expect same number of data
    new_benchmarks = loaded_db.read("benchmarks")
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert len(new_benchmarks) == 3
    assert len(new_experiments) == 3
    assert len(new_trials) == 24
    assert len(new_algos) == 3

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


def test_load_bump_no_benchmarks(empty_database):
    """Test load all database with --resolve --bump"""
    data_source = LOAD_DATA

    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {data_source}")
    benchmarks = loaded_db.read("benchmarks")
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(benchmarks) == 0
    assert len(experiments) == 3
    assert len(trials) == 24
    assert len(algos) == 3

    execute(f"db load {data_source} -r bump")
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    new_benchmarks = loaded_db.read("benchmarks")
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert len(new_benchmarks) == 0
    assert len(new_experiments) == 3 * 2
    assert len(new_trials) == 24 * 2
    assert len(new_algos) == 3 * 2

    execute(f"db load {data_source} -r bump")
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    third_benchmarks = loaded_db.read("benchmarks")
    third_experiments = loaded_db.read("experiments")
    third_trials = loaded_db.read("trials")
    third_algos = loaded_db.read("algo")
    assert len(third_benchmarks) == 0
    assert len(third_experiments) == 3 * 3
    assert len(third_trials) == 24 * 3
    assert len(third_algos) == 3 * 3


def test_load_bump_with_benchmarks(empty_database, capsys):
    """Test load all database with --resolve --bump"""
    data_source = LOAD_DATA_WITH_BENCHMARKS

    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    # First execution should pass, as destination contains nothing.
    execute(f"db load {data_source}")
    benchmarks = loaded_db.read("benchmarks")
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(benchmarks) == 3
    assert len(experiments) == 3
    assert len(trials) == 24
    assert len(algos) == 3

    # New execution should fail, as benchmarks don't currently support bump.
    execute(f"db load {data_source} -r bump", assert_code=1)
    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: Can't bump benchmark version, as benchmarks do not currently support versioning."
    )
    # Destination should have not changed.
    new_benchmarks = loaded_db.read("benchmarks")
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert new_benchmarks == benchmarks
    assert new_experiments == experiments
    assert new_trials == trials
    assert new_algos == algos


def test_load_one_experiment(empty_database):
    """Test load experiment test_single_exp"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -n test_single_exp")
    assert len(loaded_db.read("benchmarks")) == 0
    experiments = loaded_db.read("experiments")
    algos = loaded_db.read("algo")
    trials = loaded_db.read("trials")
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


def test_load_one_experiment_other_version(empty_database):
    """Test load version 1 of experiment test_single_exp"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -n test_single_exp -v 1")
    assert len(loaded_db.read("benchmarks")) == 0
    experiments = loaded_db.read("experiments")
    algos = loaded_db.read("algo")
    trials = loaded_db.read("trials")
    assert len(experiments) == 1
    (exp_data,) = experiments
    assert exp_data["name"] == "test_single_exp"
    assert exp_data["version"] == 1
    assert len(algos) == len(exp_data["algorithm"]) == 1
    # This experiment must have 12 trials (children included)
    assert len(trials) == 12
    assert all(algo["experiment"] == exp_data["_id"] for algo in algos)
    assert all(trial["experiment"] == exp_data["_id"] for trial in trials)


def test_load_one_experiment_ignore(empty_database):
    """Test load experiment test_single_exp with --resolve ignore"""

    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -n test_single_exp -v 1")
    assert len(loaded_db.read("benchmarks")) == 0
    experiments = loaded_db.read("experiments")
    algos = loaded_db.read("algo")
    trials = loaded_db.read("trials")
    assert len(experiments) == 1
    assert len(algos) == 1
    assert len(trials) == 12

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r ignore -n test_single_exp -v 1")
    new_experiments = loaded_db.read("experiments")
    new_algos = loaded_db.read("algo")
    new_trials = loaded_db.read("trials")
    assert len(new_experiments) == 1
    assert len(new_algos) == 1
    assert len(new_trials) == 12

    # IDs should have not changed
    assert len(common_indices(experiments, new_experiments)) == 1
    assert len(common_indices(algos, new_algos)) == 1
    assert len(common_indices(new_trials, trials)) == 12


def test_load_one_experiment_overwrite(empty_database):
    """Test load experiment test_single_exp with --resolve overwrite"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -n test_single_exp -v 1")
    assert len(loaded_db.read("benchmarks")) == 0
    experiments = loaded_db.read("experiments")
    algos = loaded_db.read("algo")
    trials = loaded_db.read("trials")
    assert len(experiments) == 1
    assert len(algos) == 1
    assert len(trials) == 12

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r overwrite -n test_single_exp -v 1")
    new_experiments = loaded_db.read("experiments")
    new_algos = loaded_db.read("algo")
    new_trials = loaded_db.read("trials")
    assert len(new_experiments) == 1
    assert len(new_algos) == 1
    assert len(new_trials) == 12


def test_load_one_experiment_bump(empty_database):
    """Test load experiment test_single_exp with --resolve bump"""
    storage = setup_storage()
    loaded_db = storage._db
    assert len(loaded_db.read("benchmarks")) == 0
    assert len(loaded_db.read("experiments")) == 0
    assert len(loaded_db.read("trials")) == 0
    assert len(loaded_db.read("algo")) == 0

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -n test_single_exp -v 1")
    assert len(loaded_db.read("benchmarks")) == 0
    experiments = loaded_db.read("experiments")
    trials = loaded_db.read("trials")
    algos = loaded_db.read("algo")
    assert len(experiments) == 1
    assert len(algos) == 1
    assert len(trials) == 12

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r bump -n test_single_exp -v 1")
    # Duplicated data should be bumped, so we must expect twice quantity of data.
    new_experiments = loaded_db.read("experiments")
    new_trials = loaded_db.read("trials")
    new_algos = loaded_db.read("algo")
    assert len(new_experiments) == 1 * 2
    assert len(new_algos) == 1 * 2
    assert len(new_trials) == 12 * 2

    execute(f"db load {LOAD_DATA_WITH_BENCHMARKS} -r bump -n test_single_exp -v 1")
    # Duplicated data should be bumped, so we must expect thrice quantity of data.
    third_experiments = loaded_db.read("experiments")
    third_trials = loaded_db.read("trials")
    third_algos = loaded_db.read("algo")
    assert len(third_experiments) == 1 * 3
    assert len(third_algos) == 1 * 3
    assert len(third_trials) == 12 * 3
