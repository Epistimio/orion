#!/usr/bin/env python
"""Common fixtures and utils for unittests and functional tests."""
import copy
import os
import pickle
import zlib
from collections import Counter
from tempfile import NamedTemporaryFile

import pytest
import yaml

import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.worker.trial import Trial
from orion.storage.backup import (
    _get_exp_key,
    dump_database,
    get_experiment_parent_links,
    get_experiment_root_links,
    get_trial_parent_links,
)


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.yaml")
    ) as f:
        exp_config = list(yaml.safe_load_all(f))

    for config in exp_config[0]:
        backward.populate_space(config)

    return exp_config


@pytest.fixture
def empty_database(storage):
    """Empty database"""


@pytest.fixture
def only_experiments_db(storage, exp_config):
    """Clean the database and insert only experiments."""
    for exp in exp_config[0]:
        storage.create_experiment(exp)


def ensure_deterministic_id(name, storage, version=1, update=None):
    """Change the id of experiment to its name."""
    experiment = storage.fetch_experiments({"name": name, "version": version})[0]
    algo_lock_info = storage.get_algorithm_lock_info(uid=experiment["_id"])

    storage.delete_experiment(uid=experiment["_id"])
    storage.delete_algorithm_lock(uid=experiment["_id"])

    _id = zlib.adler32(str((name, version)).encode())
    experiment["_id"] = _id

    if experiment["refers"]["parent_id"] is None:
        experiment["refers"]["root_id"] = _id

    if update is not None:
        experiment.update(update)

    storage.create_experiment(
        experiment,
        algo_locked=algo_lock_info.locked,
        algo_state=algo_lock_info.state,
        algo_heartbeat=algo_lock_info.heartbeat,
    )


# Experiments combinations fixtures
@pytest.fixture
def one_experiment(monkeypatch, storage):
    """Create an experiment without trials."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    name = "test_single_exp"
    orion.core.cli.main(
        ["hunt", "--init-only", "-n", name, "./black_box.py", "--x~uniform(0,1)"]
    )
    ensure_deterministic_id(name, storage)
    return storage.fetch_experiments({"name": name})[0]


@pytest.fixture
def one_experiment_changed_vcs(storage, one_experiment):
    """Create an experiment without trials."""
    experiment = experiment_builder.build(name=one_experiment["name"], storage=storage)

    experiment.metadata["VCS"] = {
        "type": "git",
        "is_dirty": False,
        "HEAD_sha": "new",
        "active_branch": "master",
        "diff_sha": None,
    }

    storage.update_experiment(experiment, metadata=experiment.metadata)


@pytest.fixture
def one_experiment_no_version(monkeypatch, one_experiment, storage):
    """Create an experiment without trials."""
    one_experiment["name"] = one_experiment["name"] + "-no-version"
    one_experiment.pop("version")

    def fetch_without_version(self, query, selection=None):
        if query.get("name") == one_experiment["name"] or query == {}:
            return [copy.deepcopy(one_experiment)]

        return []

    monkeypatch.setattr(type(storage), "fetch_experiments", fetch_without_version)

    return one_experiment


@pytest.fixture
def with_experiment_using_python_api(storage, monkeypatch, one_experiment):
    """Create an experiment without trials."""
    experiment = experiment_builder.build(
        name="from-python-api", space={"x": "uniform(0, 10)"}, storage=storage
    )

    return experiment


@pytest.fixture
def with_experiment_missing_conf_file(monkeypatch, one_experiment, storage, orionstate):
    """Create an experiment without trials."""
    exp = experiment_builder.build(name="test_single_exp", version=1, storage=storage)
    conf_file = "idontexist.yaml"
    exp.metadata["user_config"] = conf_file
    exp.metadata["user_args"] += ["--config", conf_file]

    orionstate.database.write("experiments", exp.configuration, query={"_id": exp.id})

    return exp


@pytest.fixture
def broken_refers(one_experiment, storage):
    """Create an experiment with broken refers."""
    ensure_deterministic_id(
        "test_single_exp", storage, update=dict(refers={"oups": "broken"})
    )


@pytest.fixture
def single_without_success(one_experiment, orionstate, storage):
    """Create an experiment without a successful trial."""
    statuses = list(Trial.allowed_stati)
    statuses.remove("completed")

    exp = experiment_builder.build(name="test_single_exp", storage=storage)
    x = {"name": "/x", "type": "real"}

    x_value = 0
    for status in statuses:
        x["value"] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x_value += 1
        orionstate.database.write("trials", trial.to_dict())


@pytest.fixture
def single_with_trials(single_without_success, orionstate, storage):
    """Create an experiment with all types of trials."""
    exp = experiment_builder.build(name="test_single_exp", storage=storage)

    x = {"name": "/x", "type": "real", "value": 100}
    results = {"name": "obj", "type": "objective", "value": 0}
    trial = Trial(experiment=exp.id, params=[x], status="completed", results=[results])
    orionstate.database.write("trials", trial.to_dict())
    return exp.configuration


@pytest.fixture
def two_experiments(monkeypatch, storage):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "-n",
            "test_double_exp",
            "./black_box.py",
            "--x~uniform(0,1)",
        ]
    )
    ensure_deterministic_id("test_double_exp", storage)

    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_double_exp",
            "--branch-to",
            "test_double_exp_child",
            "./black_box.py",
            "--x~+uniform(0,1,default_value=0)",
            "--y~+uniform(0,1,default_value=0)",
        ]
    )
    ensure_deterministic_id("test_double_exp_child", storage)


@pytest.fixture
def family_with_trials(two_experiments, orionstate):
    """Create two related experiments with all types of trials."""

    exp = experiment_builder.build(name="test_double_exp", storage=orionstate.storage)
    exp2 = experiment_builder.build(
        name="test_double_exp_child", storage=orionstate.storage
    )
    x = {"name": "/x", "type": "real"}
    y = {"name": "/y", "type": "real"}

    x_value = 0
    for status in Trial.allowed_stati:
        x["value"] = x_value
        y["value"] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x["value"] = x_value + 0.5  # To avoid duplicates
        trial2 = Trial(experiment=exp2.id, params=[x, y], status=status)
        x_value += 1
        orionstate.database.write("trials", trial.to_dict())
        orionstate.database.write("trials", trial2.to_dict())


@pytest.fixture
def unrelated_with_trials(family_with_trials, single_with_trials, orionstate):
    """Create two unrelated experiments with all types of trials."""
    exp = experiment_builder.build(
        name="test_double_exp_child", storage=orionstate.storage
    )

    orionstate.database.remove("trials", {"experiment": exp.id})
    orionstate.database.remove("experiments", {"_id": exp.id})


@pytest.fixture
def three_experiments(two_experiments, one_experiment):
    """Create a single experiment and an experiment and its child."""


@pytest.fixture
def three_experiments_with_trials(family_with_trials, single_with_trials):
    """Create three experiments, two unrelated, with all types of trials."""


@pytest.fixture
def three_experiments_family(two_experiments, storage):
    """Create three experiments, one of which is the parent of the other two."""
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_double_exp",
            "--branch-to",
            "test_double_exp_child2",
            "./black_box.py",
            "--x~+uniform(0,1,default_value=0)",
            "--z~+uniform(0,1,default_value=0)",
        ]
    )
    ensure_deterministic_id("test_double_exp_child2", storage)


@pytest.fixture
def three_family_with_trials(three_experiments_family, family_with_trials, orionstate):
    """Create three experiments, all related, two direct children, with all types of trials."""
    exp = experiment_builder.build(
        name="test_double_exp_child2", storage=orionstate.storage
    )
    x = {"name": "/x", "type": "real"}
    z = {"name": "/z", "type": "real"}

    x_value = 0
    for status in Trial.allowed_stati:
        x["value"] = x_value + 0.75  # To avoid duplicates
        z["value"] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, z], status=status)
        x_value += 1
        orionstate.database.write("trials", trial.to_dict())


@pytest.fixture
def three_experiments_family_branch(two_experiments, storage):
    """Create three experiments, each parent of the following one."""
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_double_exp_child",
            "--branch-to",
            "test_double_exp_grand_child",
            "./black_box.py",
            "--x~+uniform(0,1,default_value=0)",
            "--y~uniform(0,1,default_value=0)",
            "--z~+uniform(0,1,default_value=0)",
        ]
    )
    ensure_deterministic_id("test_double_exp_grand_child", storage)


@pytest.fixture
def three_family_branch_with_trials(
    three_experiments_family_branch, family_with_trials, orionstate
):
    """Create three experiments, all related, one child and one grandchild,
    with all types of trials.

    """
    exp = experiment_builder.build(
        name="test_double_exp_grand_child", storage=orionstate.storage
    )
    x = {"name": "/x", "type": "real"}
    y = {"name": "/y", "type": "real"}
    z = {"name": "/z", "type": "real"}

    x_value = 0
    for status in Trial.allowed_stati:
        x["value"] = x_value + 0.25  # To avoid duplicates
        y["value"] = x_value * 10
        z["value"] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, y, z], status=status)
        x_value += 1
        orionstate.database.write("trials", trial.to_dict())


@pytest.fixture
def two_experiments_same_name(one_experiment, storage):
    """Create two experiments with the same name but different versions."""
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_single_exp",
            "./black_box.py",
            "--x~uniform(0,1)",
            "--y~+normal(0,1)",
        ]
    )
    ensure_deterministic_id("test_single_exp", storage, version=2)


@pytest.fixture
def three_experiments_family_same_name(two_experiments_same_name, storage):
    """Create three experiments, two of them with the same name but different versions and one
    with a child.
    """
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_single_exp",
            "-v",
            "1",
            "-b",
            "test_single_exp_child",
            "./black_box.py",
            "--x~uniform(0,1)",
            "--y~+normal(0,1)",
        ]
    )
    ensure_deterministic_id("test_single_exp_child", storage)


@pytest.fixture
def three_experiments_branch_same_name(two_experiments_same_name, storage):
    """Create three experiments, two of them with the same name but different versions and last one
    with a child.
    """
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_single_exp",
            "-b",
            "test_single_exp_child",
            "./black_box.py",
            "--x~uniform(0,1)",
            "--y~normal(0,1)",
            "--z~+normal(0,1)",
        ]
    )
    ensure_deterministic_id("test_single_exp_child", storage)


@pytest.fixture
def three_experiments_same_name(two_experiments_same_name, storage):
    """Create three experiments with the same name but different versions."""
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_single_exp",
            "./black_box.py",
            "--x~uniform(0,1)",
            "--y~normal(0,1)",
            "--z~+normal(0,1)",
        ]
    )
    ensure_deterministic_id("test_single_exp", storage, version=3)


@pytest.fixture
def three_experiments_same_name_with_trials(
    two_experiments_same_name, orionstate, storage
):
    """Create three experiments with the same name but different versions."""

    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "test_single_exp",
            "./black_box.py",
            "--x~uniform(0,1)",
            "--y~normal(0,1)",
            "--z~+normal(0,1)",
        ]
    )
    ensure_deterministic_id("test_single_exp", storage, version=3)

    exp = experiment_builder.build(name="test_single_exp", version=1, storage=storage)
    exp2 = experiment_builder.build(name="test_single_exp", version=2, storage=storage)
    exp3 = experiment_builder.build(name="test_single_exp", version=3, storage=storage)

    x = {"name": "/x", "type": "real"}
    y = {"name": "/y", "type": "real"}
    z = {"name": "/z", "type": "real"}
    x_value = 0
    for status in Trial.allowed_stati:
        x["value"] = x_value + 0.1  # To avoid duplicates
        y["value"] = x_value * 10
        z["value"] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x], status=status)
        trial2 = Trial(experiment=exp2.id, params=[x, y], status=status)
        trial3 = Trial(experiment=exp3.id, params=[x, y, z], status=status)
        orionstate.database.write("trials", trial.to_dict())
        orionstate.database.write("trials", trial2.to_dict())
        orionstate.database.write("trials", trial3.to_dict())
        x_value += 1


@pytest.fixture
def three_experiments_branch_same_name_trials(
    three_experiments_branch_same_name, orionstate, storage
):
    """Create three experiments, two of them with the same name but different versions and one
    with a child, and add trials including children trials.

    Add algorithm state for one experiment.

    NB: It seems 2 experiments are children:
    * test_single_exp_child.1 child of test_single_exp.2
    * test_single_exp.2 child of test_single_exp.1
    * test_single_exp.1 has no parent
    """
    exp1 = experiment_builder.build(name="test_single_exp", version=1, storage=storage)
    exp2 = experiment_builder.build(name="test_single_exp", version=2, storage=storage)
    exp3 = experiment_builder.build(
        name="test_single_exp_child", version=1, storage=storage
    )

    x = {"name": "/x", "type": "real"}
    y = {"name": "/y", "type": "real"}
    z = {"name": "/z", "type": "real"}
    x_value = 0.0
    for status in Trial.allowed_stati:
        x["value"] = x_value + 0.1  # To avoid duplicates
        y["value"] = x_value * 10
        z["value"] = x_value * 100
        trial1 = Trial(experiment=exp1.id, params=[x], status=status)
        trial2 = Trial(experiment=exp2.id, params=[x, y], status=status)
        trial3 = Trial(experiment=exp3.id, params=[x, y, z], status=status)
        # Add a child to a trial from exp1
        child = trial1.branch(params={"/x": 1})
        orionstate.database.write("trials", trial1.to_dict())
        orionstate.database.write("trials", trial2.to_dict())
        orionstate.database.write("trials", trial3.to_dict())
        orionstate.database.write("trials", child.to_dict())
        x_value += 1
    # exp1 should have 12 trials (including child trials)
    # exp2 and exp3 should have 6 trials each

    # Add some algo data for exp1
    orionstate.database.read_and_write(
        collection_name="algo",
        query={"experiment": exp1.id},
        data={
            "state": pickle.dumps(
                {"my_algo_state": "some_data", "my_other_state_data": "some_other_data"}
            )
        },
    )


@pytest.fixture
def three_experiments_branch_same_name_trials_benchmarks(
    three_experiments_branch_same_name_trials, orionstate, storage
):
    """Create three experiments, two of them with the same name but different versions and one
    with a child, and add trials including children trials.

    Add algorithm state for one experiment.
    Add benchmarks to database.
    """
    # Add benchmarks, copied from db_dashboard_full.pkl
    orionstate.database.write(
        "benchmarks",
        [
            {
                "_id": 1,
                "algorithms": ["gridsearch", "random"],
                "name": "branin_baselines_webapi",
                "targets": [
                    {
                        "assess": {"AverageResult": {"repetitions": 10}},
                        "task": {"Branin": {"max_trials": 50}},
                    }
                ],
            },
            {
                "_id": 2,
                "algorithms": [
                    "gridsearch",
                    "random",
                    {"tpe": {"n_initial_points": 20}},
                ],
                "name": "all_algos_webapi",
                "targets": [
                    {
                        "assess": {"AverageResult": {"repetitions": 3}},
                        "task": {
                            "Branin": {"max_trials": 10},
                            "EggHolder": {"dim": 4, "max_trials": 20},
                            "RosenBrock": {"dim": 3, "max_trials": 10},
                        },
                    }
                ],
            },
            {
                "_id": 3,
                "algorithms": ["random", {"tpe": {"n_initial_points": 20}}],
                "name": "all_assessments_webapi_2",
                "targets": [
                    {
                        "assess": {
                            "AverageRank": {"repetitions": 3},
                            "AverageResult": {"repetitions": 3},
                            "ParallelAssessment": {
                                "executor": "joblib",
                                "n_workers": (1, 2, 4, 8),
                                "repetitions": 3,
                            },
                        },
                        "task": {
                            "Branin": {"max_trials": 10},
                            "RosenBrock": {"dim": 3, "max_trials": 10},
                        },
                    }
                ],
            },
        ],
    )


@pytest.fixture
def pkl_experiments(three_experiments_branch_same_name_trials, orionstate, storage):
    """Dump three_experiments_branch_same_name_trials to a PKL file"""
    with NamedTemporaryFile(prefix="dumped_", suffix=".pkl", delete=False) as tf:
        pkl_path = tf.name
    dump_database(storage, pkl_path, overwrite=True)
    return pkl_path


@pytest.fixture
def pkl_experiments_and_benchmarks(
    three_experiments_branch_same_name_trials_benchmarks, orionstate, storage
):
    """Dump three_experiments_branch_same_name_trials_benchmarks to a PKL file"""
    with NamedTemporaryFile(prefix="dumped_", suffix=".pkl", delete=False) as tf:
        pkl_path = tf.name
    dump_database(storage, pkl_path, overwrite=True)
    return pkl_path


@pytest.fixture
def other_empty_database():
    """Get an empty database and associated configuration file.

    To be used where we need both global config (e.g. for pkl_* fixtures)
    and another config for an empty database.
    """
    from orion.storage.base import setup_storage

    with NamedTemporaryFile(prefix="empty_", suffix=".pkl", delete=False) as tf:
        pkl_path = tf.name
    with NamedTemporaryFile(prefix="orion_config_", suffix=".yaml", delete=False) as tf:
        config_content = f"""
storage:
    database:
        type: 'pickleddb'
        host: '{pkl_path}'
""".lstrip()
        tf.write(config_content.encode())
        cfg_path = tf.name
    storage = setup_storage({"database": {"type": "pickleddb", "host": pkl_path}})
    return storage, cfg_path


class _Helpers:
    """Helper functions for testing.

    Primarily provided for tests that use fixture (and derived)
    `three_experiments_branch_same_name_trials_benchmarks`
    """

    @staticmethod
    def check_db(db, nb_exps, nb_algos, nb_trials, nb_benchmarks, nb_child_exps=0):
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

    @staticmethod
    def check_exp(
        db,
        name,
        version,
        nb_trials,
        nb_child_trials=0,
        algo_state=None,
        trial_links=None,
    ):
        """Check experiment.
        - Check if we found experiment.
        - Check if we found exactly 1 algorithm for this experiment.
        - Check algo state if algo_state is provided
        - Check if we found expected number of trials for this experiment.
        - Check if we found expecter number of child trials into experiment trials.
        - Check if we found expected trial links if provided.
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

        if trial_links is not None:
            trial_graph = get_trial_parent_links(trials)
            given_links = sorted(trial_graph.get_sorted_links())
            trial_links = sorted(trial_links)
            assert len(trial_links) == len(given_links)
            assert trial_links == given_links

    @staticmethod
    def check_empty_db(loaded_db):
        """Check that given database is empty"""
        _Helpers.check_db(
            loaded_db, nb_exps=0, nb_algos=0, nb_trials=0, nb_benchmarks=0
        )

    @staticmethod
    def assert_tested_db_structure(dumped_db, nb_orig_benchmarks=3, nb_duplicated=1):
        """Check counts and experiments for database from specific fixture
        `three_experiments_branch_same_name_trials[_benchmarks]`.
        """
        _Helpers.check_db(
            dumped_db,
            nb_exps=3 * nb_duplicated,
            nb_algos=3 * nb_duplicated,
            nb_trials=24 * nb_duplicated,
            nb_benchmarks=nb_orig_benchmarks * nb_duplicated,
            nb_child_exps=2 * nb_duplicated,
        )
        expected_parent_links = []
        expected_root_links = []
        for i in range(nb_duplicated):
            _Helpers.check_exp(
                dumped_db, "test_single_exp", 1 + 2 * i, nb_trials=12, nb_child_trials=6
            )
            _Helpers.check_exp(dumped_db, "test_single_exp", 2 + 2 * i, nb_trials=6)
            _Helpers.check_exp(dumped_db, "test_single_exp_child", 1 + i, nb_trials=6)
            expected_parent_links.extend(
                [
                    (("test_single_exp", 1 + 2 * i), ("test_single_exp", 2 + 2 * i)),
                    (("test_single_exp", 2 + 2 * i), ("test_single_exp_child", 1 + i)),
                    (("test_single_exp_child", 1 + i), None),
                ]
            )
            expected_root_links.extend(
                [
                    (("__root__",), ("test_single_exp", 1 + 2 * i)),
                    (("test_single_exp", 1 + 2 * i), ("test_single_exp", 2 + 2 * i)),
                    (("test_single_exp", 1 + 2 * i), ("test_single_exp_child", 1 + i)),
                    (("test_single_exp", 2 + 2 * i), None),
                    (("test_single_exp_child", 1 + i), None),
                ]
            )
        # Test experiments parent links.
        experiments = dumped_db.read("experiments")
        parent_graph = get_experiment_parent_links(experiments)
        assert sorted(parent_graph.get_sorted_links()) == sorted(expected_parent_links)
        # Test experiments root links.
        root_graph = get_experiment_root_links(experiments)
        root_links = sorted(root_graph.get_sorted_links())
        assert root_links == sorted(expected_root_links)
        # Check that experiment with root key (__root__,)
        # do have same root ID as experiment ID
        key_to_exp = {_get_exp_key(exp): exp for exp in experiments}
        nb_verified_identical_roots = 0
        for root_key, exp_key in root_links:
            if root_key == ("__root__",):
                exp = key_to_exp[exp_key]
                assert exp["_id"] == exp["refers"]["root_id"]
                nb_verified_identical_roots += 1
        assert nb_verified_identical_roots == 1 * nb_duplicated

    @staticmethod
    def assert_tested_trial_status(dumped_db, nb_duplicated=1, counts=None):
        """Check that trials have valid status."""
        if counts is None:
            counts = {
                "new": 9,
                "reserved": 3,
                "suspended": 3,
                "completed": 3,
                "interrupted": 3,
                "broken": 3,
            }
        trial_status_count = Counter(
            trial["status"] for trial in dumped_db.read("trials")
        )
        assert len(trial_status_count) == len(counts)
        for status, count in counts.items():
            assert trial_status_count[status] == count * nb_duplicated

    @staticmethod
    def check_unique_import(
        loaded_db,
        name,
        version,
        nb_trials,
        nb_child_trials=0,
        nb_versions=1,
        algo_state=None,
        trial_links=None,
    ):
        """Check all versions of an experiment in given database"""
        _Helpers.check_db(
            loaded_db,
            nb_exps=1 * nb_versions,
            nb_algos=1 * nb_versions,
            nb_trials=nb_trials * nb_versions,
            nb_benchmarks=0,
        )
        for i in range(nb_versions):
            _Helpers.check_exp(
                loaded_db,
                name,
                version + i,
                nb_trials=nb_trials,
                nb_child_trials=nb_child_trials,
                algo_state=algo_state,
                trial_links=trial_links,
            )

    @staticmethod
    def check_unique_import_test_single_expV1(loaded_db, nb_versions=1):
        """Check all versions of original experiment test_single_exp.1 in given database"""
        _Helpers.check_unique_import(
            loaded_db,
            "test_single_exp",
            1,
            nb_trials=12,
            nb_child_trials=6,
            nb_versions=nb_versions,
            algo_state={
                "my_algo_state": "some_data",
                "my_other_state_data": "some_other_data",
            },
            trial_links=[
                (
                    "9dbe618878008376d0ef47dba77b4175",
                    "7bc7d88c3f84329ae15667af1fc5eba0",
                ),
                ("7bc7d88c3f84329ae15667af1fc5eba0", None),
                (
                    "68e541fa91d9017a50fe534c2e70e34c",
                    "0caeb769dd8becc1c5064d3638128948",
                ),
                ("0caeb769dd8becc1c5064d3638128948", None),
                (
                    "ebd7c227cd7d1911c3b56daa9d02b2c2",
                    "0e6dce570d2bec70b0c7e26ba6aab617",
                ),
                ("0e6dce570d2bec70b0c7e26ba6aab617", None),
                (
                    "26da495bc13561b163e1e67654c913d4",
                    "7fbcacb8b1a6fd12d57f8b84de009c42",
                ),
                ("7fbcacb8b1a6fd12d57f8b84de009c42", None),
                (
                    "284af14179121d0e8df8e7fc856f5920",
                    "a40d030ff08ebbb7d97ecffaf93fe1f6",
                ),
                ("a40d030ff08ebbb7d97ecffaf93fe1f6", None),
                (
                    "938087683a168d4640ee3f72942d2d16",
                    "44dc1dd034b0dddca891847b8aac31fb",
                ),
                ("44dc1dd034b0dddca891847b8aac31fb", None),
            ],
        )

    @staticmethod
    def check_unique_import_test_single_expV2(loaded_db, nb_versions=1):
        """Check all versions of original experiment test_single_exp.2 in given database"""
        _Helpers.check_unique_import(
            loaded_db, "test_single_exp", 2, nb_trials=6, nb_versions=nb_versions
        )


@pytest.fixture
def testing_helpers():
    return _Helpers
