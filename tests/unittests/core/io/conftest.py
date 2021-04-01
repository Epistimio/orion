#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for io tests."""


import copy
import os

import pytest

from orion.core.evc import conflicts


@pytest.fixture()
def config_file():
    """Open config file with new config"""
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "orion_config.yaml"
    )

    return open(file_path)


@pytest.fixture()
def old_config_file():
    """Open config file with original config from an experiment in db"""
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "orion_old_config.yaml"
    )

    return open(file_path)


@pytest.fixture()
def incomplete_config_file():
    """Open config file with partial database configuration"""
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "orion_incomplete_config.yaml"
    )

    return open(file_path)


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
