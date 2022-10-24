#!/usr/bin/env python
"""Common fixtures and utils for io tests."""


import copy
import os
import tempfile

import pytest

from orion.core.evc import conflicts


class _GenerateConfig:
    def __init__(self, name) -> None:
        self.name = name
        self.generated_config = None
        self.database_file = None

    def __enter__(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.name)

        self.generated_config = tempfile.NamedTemporaryFile()
        self.database_file = tempfile.NamedTemporaryFile()

        with open(file_path) as config:
            new_config = config.read().replace("${FILE}", self.database_file.name)
            self.generated_config.write(new_config.encode("utf-8"))
            self.generated_config.flush()

        return open(self.generated_config.name)

    def __exit__(self, *args, **kwargs):
        self.database_file.close()
        self.generated_config.close()


@pytest.fixture()
def raw_config():
    """Open config file with new config"""
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "orion_config.yaml"
    )
    return open(file_path)


@pytest.fixture()
def config_file():
    """Open config file with new config"""
    with _GenerateConfig("orion_config.yaml") as file:
        yield file


@pytest.fixture()
def old_config_file():
    """Open config file with original config from an experiment in db"""
    with _GenerateConfig("orion_old_config.yaml") as file:
        yield file


@pytest.fixture()
def incomplete_config_file():
    """Open config file with partial database configuration"""
    with _GenerateConfig("orion_incomplete_config.yaml") as file:
        yield file


@pytest.fixture
def parent_config():
    """Generate a new experiment configuration"""
    return dict(_id="test", name="test", metadata={"user": "corneauf"}, version=1)


@pytest.fixture
def child_config(parent_config):
    """Generate a new experiment configuration"""
    config = copy.deepcopy(parent_config)
    config["_id"] = "test2"
    config["refers"] = {"parent_id": "test"}
    config["version"] = 2

    return config


@pytest.fixture
def experiment_name_conflict(storage, parent_config, child_config):
    """Generate an experiment name conflict"""
    exps = storage.fetch_experiments({"name": "test"}) + storage.fetch_experiments(
        {"name": "test2"}
    )
    for exp in exps:
        storage.delete_experiment(uid=exp["_id"])
    storage.create_experiment(parent_config)
    storage.create_experiment(child_config)
    return conflicts.ExperimentNameConflict(parent_config, parent_config)
