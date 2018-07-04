#!/usr/bin/env python
"""Common fixtures and utils for evc unit-tests."""
import copy

import pytest

from orion.core.evc import conflicts
from orion.core.io.space_builder import DimensionBuilder


@pytest.fixture
def new_config():
    """Generate a new experiment configuration"""
    return dict(
        name='test',
        algorithms='fancy',
        metadata={'hash_commit': 'new',
                  'user_script': 'abs_path/black_box.py',
                  'user_args':
                  ['--new~normal(0,2)', '--changed~normal(0,2)'],
                  'user': 'some_user_name'})


@pytest.fixture
def old_config(create_db_instance):
    """Generate an old experiment configuration"""
    config = dict(
        name='test',
        algorithms='random',
        metadata={'hash_commit': 'old',
                  'user_script': 'abs_path/black_box.py',
                  'user_args':
                  ['--missing~uniform(-10,10)', '--changed~uniform(-10,10)'],
                  'user': 'some_user_name'})

    create_db_instance.write('experiments', config)
    return config


@pytest.fixture
def new_dimension_conflict(old_config, new_config):
    """Generate a new dimension conflict for new experiment configuration"""
    name = 'new'
    prior = 'normal(0, 2)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_with_default_conflict(old_config, new_config):
    """Generate a new dimension conflict with default value for new experiment configuration"""
    name = 'new'
    prior = 'normal(0, 2, default_value=0.001)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_same_prior_conflict(old_config, new_config):
    """Generate a new dimension conflict with different prior for renaming tests"""
    name = 'new'
    prior = 'uniform(-10, 10)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def changed_dimension_conflict(old_config, new_config):
    """Generate a changed dimension conflict"""
    name = 'changed'
    old_prior = 'uniform(-10, 10)'
    new_prior = 'normal(0, 2)'
    dimension = DimensionBuilder().build(name, old_prior)
    return conflicts.ChangedDimensionConflict(old_config, new_config,
                                              dimension, old_prior, new_prior)


@pytest.fixture
def missing_dimension_conflict(old_config, new_config):
    """Generate a missing dimension conflict"""
    name = 'missing'
    prior = 'uniform(-10, 10)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def missing_dimension_with_default_conflict(old_config, new_config):
    """Generate a missing dimension conflict with a default value"""
    name = 'missing'
    prior = 'uniform(-10, 10, default_value=0.0)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def algorithm_conflict(old_config, new_config):
    """Generate an algorithm configuration conflict"""
    return conflicts.AlgorithmConflict(old_config, new_config)


@pytest.fixture
def code_conflict(old_config, new_config):
    """Generate a code conflict"""
    return conflicts.CodeConflict(old_config, new_config)


@pytest.fixture
def cli_conflict(old_config, new_config):
    """Generate a commandline conflict"""
    new_config = copy.deepcopy(new_config)
    new_config['metadata']['user_args'].append("--some-new=args")
    return conflicts.CommandLineConflict(old_config, new_config)


@pytest.fixture
def config_conflict(old_config, new_config):
    """Generate a script config conflict"""
    return conflicts.ScriptConfigConflict(old_config, new_config)


@pytest.fixture
def experiment_name_conflict(old_config, new_config):
    """Generate an experiment name conflict"""
    return conflicts.ExperimentNameConflict(old_config, new_config)
