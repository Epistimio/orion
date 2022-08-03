#!/usr/bin/env python
"""Common fixtures and utils for unittests and functional tests."""
import copy
import os
import zlib

import pytest
import yaml

import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.worker.trial import Trial


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
def only_experiments_db(storage, exp_config):
    """Clean the database and insert only experiments."""
    for exp in exp_config[0]:
        storage.create_experiment(exp)


def ensure_deterministic_id(name, storage, version=1, update=None):
    """Change the id of experiment to its name."""
    experiment = storage.fetch_experiments({"name": name, "version": version})[0]
    storage.delete_experiment(uid=experiment["_id"])
    _id = zlib.adler32(str((name, version)).encode())
    experiment["_id"] = _id

    if experiment["refers"]["parent_id"] is None:
        experiment["refers"]["root_id"] = _id

    if update is not None:
        experiment.update(update)

    storage.create_experiment(experiment)


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
