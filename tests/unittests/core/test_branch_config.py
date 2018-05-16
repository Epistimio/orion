#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.experiment_branch_builder`."""
import pytest

from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder

@pytest.fixture
def parent_config():
    """Create a configuration that will not hit the database."""
    config = dict(
        name='test',
        metadata={'user_script': 'abs_path/black_box.py',
                  'user_args': 
                        ['-x~uniform(0,1)', '-y~normal(0,1)', '-z~uniform(0,10)']})
    return config


@pytest.fixture
def child_config():
    """Create a child branching from the test experiment"""
    config = dict(
        name='test2',
        metadata={'user_script': 'abs_path/black_box.py',
                  'user_args': 
                        ['-x~uniform(0,1)', '-y~normal(0,1)', '-z~uniform(0,10)']})
    return config


@pytest.fixture
def missing_config(child_config):
    del(child_config['metadata']['user_args'][0])
    return child_config


@pytest.fixture
def new_config(child_config):
    child_config['metadata']['user_args'].append('-w_d~normal(0,1)')
    return child_config


@pytest.fixture
def changed_config(child_config):
    second_element = child_config['metadata']['user_args'][1]
    second_element = second_element.replace('normal', 'uniform')
    child_config['metadata']['user_args'][1] = second_element
    return child_config


def test_no_conflicts(parent_config, child_config):
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert len(branch_builder.conflicts) == 0
    assert branch_builder.experiment_space == branch_builder.conflicting_space


def test_missing_conflict(parent_config, missing_config):
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/x'
    assert conflict.status == 'missing'


def test_new_conflict(parent_config, new_config):
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/w_d'
    assert conflict.status == 'new'


def test_changed_conflict(parent_config, changed_config):
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/y'
    assert conflict.status == 'changed'


def test_add_new(parent_config, new_config):
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('w_d')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_add_changed(parent_config, changed_config):
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)
    branch_builder.add_dimension('y')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_remove_missing(parent_config, missing_config):
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)
    branch_builder.remove_dimension('x')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['remove']) == 1


def test_rename_missing(parent_config, missing_config):
    missing_config['metadata']['user_args'].append('-w_d~normal(0,1)')
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)
    branch_builder.rename_dimension(['x', 'w_d'])

    assert len(branch_builder.conflicts) == 2
    
    conflicts = branch_builder.conflicts

    for conflict in conflicts:
        assert conflict.is_solved is True
        if conflict.dimension.name == '/x':
            assert conflict.status == 'missing'
        else:
            assert conflict.status == 'new'

    operations = branch_builder.operations['rename']
    assert len(operations) == 1
    old, new = operations[0]

    assert old in conflicts
    assert new in conflicts


def test_reset_dimension(parent_config, new_config):
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)

    branch_builder.add_dimension('w_d')
    branch_builder.reset_dimension('w_d')

    conflict = branch_builder.conflicts[0]

    assert conflict.status == 'new'
    assert conflict.is_solved is False
    assert len(branch_builder.operations['add']) == 0


def test_dimension_conflict(parent_config, new_config):
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    
    assert branch_builder.get_dimension_conflict('w_d').dimension.name == '/w_d'


def test_name_experiment(parent_config, child_config):
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert branch_builder.conflicting_config['name'] == 'test2'
    branch_builder.change_experiment_name('test3')
    assert branch_builder.conflicting_config['name'] == 'test3'


def test_bad_name_experiment(parent_config, child_config):
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert branch_builder.experiment_config['name'] == 'test'
    branch_builder.change_experiment_name('test')
    assert branch_builder.conflicting_config['name'] == 'test2'
