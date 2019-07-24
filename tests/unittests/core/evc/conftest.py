#!/usr/bin/env python
"""Common fixtures and utils for evc unit-tests."""

import pytest

from orion.core.evc import conflicts


@pytest.fixture
def parent_new_config():
    """Generate a new experiment configuration"""
    return dict(
        _id='test',
        name='test',
        metadata={'user': 'corneauf'},
        version=1)


@pytest.fixture
def parent_old_config(create_db_instance):
    """Generate an old experiment configuration"""
    config = dict(
        _id='test',
        name='test',
        metadata={'user': 'corneauf'},
        version=1)

    create_db_instance.write('experiments', config)
    return config


@pytest.fixture
def child_new_config():
    """Generate a new experiment configuration"""
    return dict(
        _id="test2",
        name='test',
        version=2,
        metadata={'user': 'corneauf'},
        refers={'parent_id': 'test'})


@pytest.fixture
def child_old_config(create_db_instance):
    """Generate an old experiment configuration"""
    config = dict(
        _id="test2",
        name='test',
        version=2,
        metadata={'user': 'corneauf'},
        refers={'parent_id': 'test'})

    create_db_instance.write('experiments', config)
    return config


@pytest.fixture
def exp_no_child_conflict(parent_old_config, parent_new_config):
    """Generate an experiment name conflict"""
    return conflicts.ExperimentNameConflict(parent_old_config, parent_new_config)
