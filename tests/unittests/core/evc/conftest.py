#!/usr/bin/env python
"""Common fixtures and utils for evc unit-tests."""
import copy

import pytest

from orion.core.evc import conflicts


@pytest.fixture
def parent_config():
    """Generate a new experiment configuration"""
    return dict(
        _id='test',
        name='test',
        metadata={'user': 'corneauf'},
        version=1)


@pytest.fixture
def child_config(parent_config):
    """Generate a new experiment configuration"""
    config = copy.deepcopy(parent_config)
    config['_id'] = "test2"
    config['refers'] = {'parent_id': 'test'}
    config['version'] = 2

    return config


@pytest.fixture
def exp_no_child_conflict(create_db_instance, parent_config):
    """Generate an experiment name conflict"""
    create_db_instance.write('experiments', parent_config)
    return conflicts.ExperimentNameConflict(parent_config, parent_config)


@pytest.fixture
def exp_w_child_conflict(create_db_instance, parent_config, child_config):
    """Generate an experiment name conflict"""
    create_db_instance.write('experiments', parent_config)
    create_db_instance.write('experiments', child_config)
    return conflicts.ExperimentNameConflict(child_config, child_config)


@pytest.fixture
def exp_w_child_as_parent_conflict(create_db_instance, parent_config, child_config):
    """Generate an experiment name conflict"""
    create_db_instance.write('experiments', parent_config)
    create_db_instance.write('experiments', child_config)
    return conflicts.ExperimentNameConflict(parent_config, parent_config)


@pytest.fixture
def existing_exp_conflict(create_db_instance, parent_config):
    """Generate an experiment name conflict"""
    create_db_instance.write('experiments', parent_config)
    create_db_instance.write('experiments', {'name': 'dummy', 'version': 1})
    return conflicts.ExperimentNameConflict(parent_config, parent_config)
