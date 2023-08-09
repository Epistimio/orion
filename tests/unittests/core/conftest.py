#!/usr/bin/env python
"""Common fixtures and utils for tests."""

import copy
import getpass
import os

import pytest
import yaml

import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.algo.space import Categorical, Integer, Real, Space
from orion.core.evc import conflicts
from orion.core.io.convert import JSONConverter, YAMLConverter
from orion.core.io.space_builder import DimensionBuilder
from orion.core.utils import format_trials
from orion.core.worker.trial import Trial
from orion.testing import MockDatetime

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_SAMPLE = os.path.join(TEST_DIR, "sample_config.yml")
YAML_DIFF_SAMPLE = os.path.join(TEST_DIR, "sample_config_diff.yml")
JSON_SAMPLE = os.path.join(TEST_DIR, "sample_config.json")
UNKNOWN_SAMPLE = os.path.join(TEST_DIR, "sample_config.txt")
UNKNOWN_TEMPLATE = os.path.join(TEST_DIR, "sample_config_template.txt")


@pytest.fixture(scope="session")
def yaml_sample_path():
    """Return path with a yaml sample file."""
    return os.path.abspath(YAML_SAMPLE)


@pytest.fixture(scope="session")
def yaml_diff_sample_path():
    """Return path with a different yaml sample file."""
    return os.path.abspath(YAML_DIFF_SAMPLE)


@pytest.fixture
def yaml_config(yaml_sample_path):
    """Return a list containing the key and the sample path for a yaml config."""
    return ["--config", yaml_sample_path]


@pytest.fixture
def yaml_diff_config(yaml_diff_sample_path):
    """Return a list containing the key and the sample path for a different yaml config."""
    return ["--config", yaml_diff_sample_path]


@pytest.fixture(scope="session")
def json_sample_path():
    """Return path with a json sample file."""
    return JSON_SAMPLE


@pytest.fixture
def json_config(json_sample_path):
    """Return a list containing the key and the sample path for a json config."""
    return ["--config", json_sample_path]


@pytest.fixture(scope="session")
def unknown_type_sample_path():
    """Return path with a sample file of unknown configuration filetype."""
    return UNKNOWN_SAMPLE


@pytest.fixture(scope="session")
def unknown_type_template_path():
    """Return path with a template file of unknown configuration filetype."""
    return UNKNOWN_TEMPLATE


@pytest.fixture(scope="session")
def some_sample_path():
    """Return path with a sample file of unknown configuration filetype."""
    return os.path.join(TEST_DIR, "some_sample_config.txt")


@pytest.fixture
def some_sample_config(some_sample_path):
    """Return a list containing the key and the sample path for some config."""
    return ["--config", some_sample_path]


@pytest.fixture(scope="session")
def yaml_converter():
    """Return a yaml converter."""
    return YAMLConverter()


@pytest.fixture(scope="session")
def json_converter():
    """Return a json converter."""
    return JSONConverter()


@pytest.fixture(scope="module")
def space():
    """Construct a simple space with every possible kind of Dimension."""
    space = Space()
    categories = {"asdfa": 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    dim = Categorical("yolo", categories, shape=2)
    space.register(dim)
    dim = Integer("yolo2", "uniform", -3, 6)
    space.register(dim)
    dim = Real("yolo3", "alpha", 0.9)
    space.register(dim)
    return space


@pytest.fixture(scope="module")
def hierarchical_space():
    """Construct a space with hierarchical Dimensions."""
    space = Space()
    categories = {"asdfa": 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    dim = Categorical("yolo.first", categories, shape=2)
    space.register(dim)
    dim = Integer("yolo.second", "uniform", -3, 6)
    space.register(dim)
    dim = Real("yoloflat", "alpha", 0.9)
    space.register(dim)
    return space


@pytest.fixture(scope="function")
def fixed_suggestion_value(space):
    """Return the same trial from a possible space."""
    return (("asdfa", 2), 0, 3.5)


@pytest.fixture(scope="function")
def fixed_suggestion(fixed_suggestion_value, space):
    """Return the same trial from a possible space."""
    return format_trials.tuple_to_trial(fixed_suggestion_value, space)


@pytest.fixture()
def with_user_tsirif(monkeypatch):
    """Make ``getpass.getuser()`` return ``'tsirif'``."""
    monkeypatch.setattr(getpass, "getuser", lambda: "tsirif")


@pytest.fixture()
def with_user_bouthilx(monkeypatch):
    """Make ``getpass.getuser()`` return ``'bouthilx'``."""
    monkeypatch.setattr(getpass, "getuser", lambda: "bouthilx")


@pytest.fixture()
def with_user_dendi(monkeypatch):
    """Make ``getpass.getuser()`` return ``'dendi'``."""
    monkeypatch.setattr(getpass, "getuser", lambda: "dendi")


dendi_exp_config = dict(
    name="supernaedo2-dendi",
    space={
        "/decoding_layer": "choices(['rnn', 'lstm_with_attention', 'gru'])",
        "/encoding_layer": "choices(['rnn', 'lstm', 'gru'])",
    },
    metadata={
        "user": "dendi",
        "orion_version": "XYZ",
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    pool_size=1,
    max_trials=1000,
    working_dir="",
    algorithm={"dumbalgo": {}},
    producer={"strategy": "NoParallelStrategy"},
)


dendi_base_trials = [
    {
        "status": "completed",
        "worker": 12512301,
        "submit_time": MockDatetime(2017, 11, 22, 23),
        "start_time": None,
        "end_time": MockDatetime(2017, 11, 22, 23),
        "results": [{"name": None, "type": "objective", "value": 3}],
        "params": [
            {"name": "/decoding_layer", "type": "categorical", "value": "rnn"},
            {"name": "/encoding_layer", "type": "categorical", "value": "lstm"},
        ],
        "parent": None,
    },
    {
        "status": "completed",
        "worker": 23415151,
        "submit_time": MockDatetime(2017, 11, 23, 0),
        "start_time": None,
        "end_time": MockDatetime(2017, 11, 23, 0),
        "results": [
            {"name": "yolo", "type": "objective", "value": 10},
            {"name": "contra", "type": "constraint", "value": 1.2},
            {"name": "naedw_grad", "type": "gradient", "value": [5, 3]},
        ],
        "params": [
            {
                "name": "/decoding_layer",
                "type": "categorical",
                "value": "lstm_with_attention",
            },
            {"name": "/encoding_layer", "type": "categorical", "value": "gru"},
        ],
        "parent": None,
    },
    {
        "status": "completed",
        "worker": 1251231,
        "submit_time": MockDatetime(2017, 11, 22, 23),
        "start_time": None,
        "end_time": MockDatetime(2017, 11, 22, 22),
        "results": [
            {"name": None, "type": "objective", "value": 2},
            {"name": "naedw_grad", "type": "gradient", "value": [-0.1, 2]},
        ],
        "params": [
            {"name": "/decoding_layer", "type": "categorical", "value": "rnn"},
            {"name": "/encoding_layer", "type": "categorical", "value": "rnn"},
        ],
        "parent": None,
    },
    {
        "status": "new",
        "worker": None,
        "submit_time": MockDatetime(2017, 11, 23, 1),
        "start_time": None,
        "end_time": None,
        "results": [{"name": None, "type": "objective", "value": None}],
        "params": [
            {"name": "/decoding_layer", "type": "categorical", "value": "rnn"},
            {"name": "/encoding_layer", "type": "categorical", "value": "gru"},
        ],
        "parent": None,
    },
    {
        "status": "new",
        "worker": None,
        "submit_time": MockDatetime(2017, 11, 23, 2),
        "start_time": None,
        "end_time": None,
        "results": [{"name": None, "type": "objective", "value": None}],
        "params": [
            {
                "name": "/decoding_layer",
                "type": "categorical",
                "value": "lstm_with_attention",
            },
            {"name": "/encoding_layer", "type": "categorical", "value": "rnn"},
        ],
        "parent": None,
    },
    {
        "status": "interrupted",
        "worker": None,
        "submit_time": MockDatetime(2017, 11, 23, 3),
        "start_time": MockDatetime(2017, 11, 23, 3),
        "end_time": None,
        "results": [{"name": None, "type": "objective", "value": None}],
        "params": [
            {
                "name": "/decoding_layer",
                "type": "categorical",
                "value": "lstm_with_attention",
            },
            {"name": "/encoding_layer", "type": "categorical", "value": "lstm"},
        ],
        "parent": None,
    },
    {
        "status": "suspended",
        "worker": None,
        "submit_time": MockDatetime(2017, 11, 23, 4),
        "start_time": MockDatetime(2017, 11, 23, 4),
        "end_time": None,
        "results": [{"name": None, "type": "objective", "value": None}],
        "params": [
            {"name": "/decoding_layer", "type": "categorical", "value": "gru"},
            {"name": "/encoding_layer", "type": "categorical", "value": "lstm"},
        ],
        "parent": None,
    },
]


@pytest.fixture()
def hacked_exp(with_user_dendi, random_dt, storage):
    """Return an `Experiment` instance to find trials in fake database."""
    storage.create_experiment(dendi_exp_config)
    exp = experiment_builder.build(name=dendi_exp_config["name"])
    storage._db.write(
        "trials",
        [Trial(experiment=exp.id, **trial).to_dict() for trial in dendi_base_trials],
    )
    return exp


###
# Fixtures for EVC tests using conflicts, present in both ./evc and ./io.
# Note: Refactoring the EVC out of orion's core should take care of getting those
#       fixtures out of general conftest.py
###


@pytest.fixture
def new_config():
    """Generate a new experiment configuration"""
    user_script = "tests/functional/demo/black_box.py"
    config = dict(
        name="test",
        algorithm="fancy",
        version=1,
        metadata={
            "VCS": "to be changed",
            "user_script": user_script,
            "user_args": [user_script, "--new~normal(0,2)", "--changed~normal(0,2)"],
            "user": "some_user_name",
            "orion_version": "UVW",
        },
    )

    backward.populate_space(config)

    return config


@pytest.fixture
def old_config_with_script_conf(old_config, tmp_path):
    """Generate a old experiment configuration with a config file"""

    old_config = copy.deepcopy(old_config)

    config_path = tmp_path / "old_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"config-hp": "uniform(0, 10)", "dropped": "uniform(-1, 5)"}, f)
    old_config["metadata"]["user_args"] += ["--config", str(config_path)]

    backward.populate_space(old_config, force_update=True)

    return old_config


@pytest.fixture
def new_config_with_script_conf(new_config, tmp_path):
    """Generate a new experiment configuration with a different config file"""

    new_config = copy.deepcopy(new_config)

    config_path = tmp_path / "new_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"config-hp": "uniform(0, 5)", "dropped": {"hp": "value"}}, f)

    new_config["metadata"]["user_args"] += ["--config", str(config_path)]

    backward.populate_space(new_config, force_update=True)

    return new_config


@pytest.fixture
def old_config(storage):
    """Generate an old experiment configuration"""
    user_script = "tests/functional/demo/black_box.py"
    config = dict(
        name="test",
        algorithm="random",
        version=1,
        metadata={
            "VCS": {
                "type": "git",
                "is_dirty": False,
                "HEAD_sha": "test",
                "active_branch": None,
                "diff_sha": "diff",
            },
            "user_script": user_script,
            "user_args": [
                user_script,
                "--missing~uniform(-10,10)",
                "--changed~uniform(-10,10)",
            ],
            "user": "some_user_name",
            "orion_version": "XYZ",
        },
    )

    backward.populate_space(config)

    storage.create_experiment(config)
    return config


@pytest.fixture
def new_dimension_conflict(old_config, new_config):
    """Generate a new dimension conflict for new experiment configuration"""
    name = "new"
    prior = "normal(0, 2)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_with_default_conflict(old_config, new_config):
    """Generate a new dimension conflict with default value for new experiment configuration"""
    name = "new"
    prior = "normal(0, 2, default_value=0.001)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_same_prior_conflict(old_config, new_config):
    """Generate a new dimension conflict with different prior for renaming tests"""
    name = "new"
    prior = "uniform(-10, 10)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def changed_dimension_conflict(old_config, new_config):
    """Generate a changed dimension conflict"""
    name = "changed"
    old_prior = "uniform(-10, 10)"
    new_prior = "normal(0, 2)"
    dimension = DimensionBuilder().build(name, old_prior)
    return conflicts.ChangedDimensionConflict(
        old_config, new_config, dimension, old_prior, new_prior
    )


@pytest.fixture
def changed_dimension_shape_conflict(old_config, new_config):
    """Generate a changed shape dimension conflict"""
    name = "changed_shape"
    old_prior = "uniform(-10, 10)"
    new_prior = "uniform(-10, 10, shape=2)"
    dimension = DimensionBuilder().build(name, old_prior)
    return conflicts.ChangedDimensionConflict(
        old_config, new_config, dimension, old_prior, new_prior
    )


@pytest.fixture
def missing_dimension_conflict(old_config, new_config):
    """Generate a missing dimension conflict"""
    name = "missing"
    prior = "uniform(-10, 10)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def missing_dimension_from_config_conflict(
    old_config_with_script_conf, new_config_with_script_conf
):
    """Generate a missing dimension conflict in the config file"""
    name = "dropped"
    prior = "uniform(-1, 5)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(
        old_config_with_script_conf, new_config_with_script_conf, dimension, prior
    )


@pytest.fixture
def missing_dimension_with_default_conflict(old_config, new_config):
    """Generate a missing dimension conflict with a default value"""
    name = "missing"
    prior = "uniform(-10, 10, default_value=0.0)"
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def algorithm_conflict(old_config, new_config):
    """Generate an algorithm configuration conflict"""
    return conflicts.AlgorithmConflict(old_config, new_config)


@pytest.fixture
def orion_version_conflict(old_config, new_config):
    """Generate an orion version conflict"""
    return conflicts.OrionVersionConflict(old_config, new_config)


@pytest.fixture
def code_conflict(old_config, new_config):
    """Generate a code conflict"""
    return conflicts.CodeConflict(old_config, new_config)


@pytest.fixture
def cli_conflict(old_config, new_config):
    """Generate a commandline conflict"""
    new_config = copy.deepcopy(new_config)
    new_config["metadata"]["user_args"].append("--some-new=args")
    new_config["metadata"]["user_args"].append("--bool-arg")
    backward.populate_space(new_config, force_update=True)
    return conflicts.CommandLineConflict(old_config, new_config)


@pytest.fixture
def config_conflict(old_config_with_script_conf, new_config_with_script_conf):
    """Generate a script config conflict"""
    return conflicts.ScriptConfigConflict(
        old_config_with_script_conf, new_config_with_script_conf
    )


@pytest.fixture
def experiment_name_conflict(old_config, new_config):
    """Generate an experiment name conflict"""
    return conflicts.ExperimentNameConflict(old_config, new_config)


@pytest.fixture
def bad_exp_parent_config():
    """Generate a new experiment configuration"""
    config = dict(
        _id="test",
        name="test",
        metadata={
            "user": "corneauf",
            "user_args": ["--x~normal(0,1)"],
            "user_script": "tests/functional/demo/black_box.py",
            "orion_version": "XYZ",
        },
        version=1,
        algorithm="random",
    )

    backward.populate_space(config)

    return config


@pytest.fixture
def bad_exp_child_config(bad_exp_parent_config):
    """Generate a new experiment configuration"""
    config = copy.deepcopy(bad_exp_parent_config)
    config["_id"] = "test2"
    config["refers"] = {"parent_id": "test"}
    config["version"] = 2

    return config
