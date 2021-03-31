#!/usr/bin/env python
"""Common fixtures and utils for evc unit-tests."""
import copy

import pytest

from orion.core.evc import conflicts


@pytest.fixture
def parent_config():
    """Generate a new experiment configuration"""
    return dict(_id=1234, name="test", metadata={"user": "corneauf"}, version=1)


@pytest.fixture
def child_config(parent_config):
    """Generate a new experiment configuration"""
    config = copy.deepcopy(parent_config)
    config["_id"] = 1235
    config["refers"] = {"parent_id": 1234}
    config["version"] = 2

    return config


@pytest.fixture
def exp_no_child_conflict(storage, parent_config):
    """Generate an experiment name conflict"""
    storage.create_experiment(parent_config)
    return conflicts.ExperimentNameConflict(parent_config, parent_config)


@pytest.fixture
def exp_w_child_conflict(storage, parent_config, child_config):
    """Generate an experiment name conflict"""
    storage.create_experiment(parent_config)
    storage.create_experiment(child_config)
    return conflicts.ExperimentNameConflict(child_config, child_config)


@pytest.fixture
def exp_w_child_as_parent_conflict(storage, parent_config, child_config):
    """Generate an experiment name conflict"""
    storage.create_experiment(parent_config)
    storage.create_experiment(child_config)
    return conflicts.ExperimentNameConflict(parent_config, parent_config)


@pytest.fixture
def existing_exp_conflict(storage, parent_config):
    """Generate an experiment name conflict"""
    storage.create_experiment(parent_config)
    storage.create_experiment({"name": "dummy", "version": 1})
    return conflicts.ExperimentNameConflict(parent_config, parent_config)
