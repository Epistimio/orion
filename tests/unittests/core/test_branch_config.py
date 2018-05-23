#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.experiment_branch_builder`."""
import pytest

import orion.core.evc.adapters as Adapters
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder


def filter_true(c):
    """Filter solved conflicts"""
    return c.is_solved is True


def filter_false(c):
    """Filter unsolved conflicts"""
    return not filter_true(c)


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
    """Create a child config with a missing dimension"""
    del(child_config['metadata']['user_args'][0])
    return child_config


@pytest.fixture
def new_config(child_config):
    """Create a child config with a new dimension"""
    child_config['metadata']['user_args'].append('-w_d~normal(0,1)')
    return child_config


@pytest.fixture
def changed_config(child_config):
    """Create a child config with a changed dimension"""
    second_element = child_config['metadata']['user_args'][1]
    second_element = second_element.replace('normal', 'uniform')
    child_config['metadata']['user_args'][1] = second_element
    return child_config


@pytest.fixture
def cl_config():
    """Create a child config with markers for commandline solving"""
    config = dict(
        name='test2',
        metadata={'user_script': 'abs_path/black_box.py',
                  'user_args':
                  ['-x~>w_d', '-y~+uniform(0,1)', '-w_d~+normal(0,1)', '-z~-']})
    return config


def test_no_conflicts(parent_config, child_config):
    """Test the case where the child is the same as the parent"""
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert len(branch_builder.conflicts) == 0
    assert branch_builder.experiment_space == branch_builder.conflicting_space


def test_missing_conflict(parent_config, missing_config):
    """Test if missing dimension is currently detected"""
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/x'
    assert conflict.status == 'missing'


def test_new_conflict(parent_config, new_config):
    """Test if new dimension is currently detected"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/w_d'
    assert conflict.status == 'new'


def test_changed_conflict(parent_config, changed_config):
    """Test if changed dimension is currently detected"""
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)

    assert len(branch_builder.conflicts) == 1
    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is False
    assert conflict.dimension.name == '/y'
    assert conflict.status == 'changed'


def test_add_single_hit(parent_config, new_config):
    """Test if adding a dimension only touches the correct statuses"""
    del new_config['metadata']['user_args'][0]
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('w_d')

    assert len(branch_builder.conflicts) == 2
    assert len(list(branch_builder.filter_conflicts(filter_true))) == 1
    assert len(list(branch_builder.filter_conflicts(filter_false))) == 1


def test_add_new(parent_config, new_config):
    """Test if adding a new dimension solves the conflict"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('w_d')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_add_changed(parent_config, changed_config):
    """Test if adding a changed dimension solves the conflict"""
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)
    branch_builder.add_dimension('y')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_remove_missing(parent_config, missing_config):
    """Test if removing a missing dimension solves the conflict"""
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)
    branch_builder.remove_dimension('x')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['remove']) == 1


def test_rename_missing(parent_config, missing_config):
    """Test if renaming a dimension to another solves both conflicts"""
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
    """Test if resetting a dimension unsolves the conflict"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)

    branch_builder.add_dimension('w_d')
    branch_builder.reset_dimension('w_d')

    conflict = branch_builder.conflicts[0]

    assert conflict.status == 'new'
    assert conflict.is_solved is False
    assert len(branch_builder.operations['add']) == 0


def test_dimension_conflict(parent_config, new_config):
    """Test if getting a dimension conflict returns it correctly"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)

    assert branch_builder.get_dimension_conflict('w_d').dimension.name == '/w_d'


def test_name_experiment(parent_config, child_config):
    """Test if changing the experiment names work for valid name"""
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert branch_builder.conflicting_config['name'] == 'test2'
    branch_builder.change_experiment_name('test3')
    assert branch_builder.conflicting_config['name'] == 'test3'


def test_bad_name_experiment(parent_config, child_config):
    """Test if changing the experiment names does not work for invalid name"""
    branch_builder = ExperimentBranchBuilder(parent_config, child_config)

    assert branch_builder.experiment_config['name'] == 'test'
    branch_builder.change_experiment_name('test')
    assert branch_builder.conflicting_config['name'] == 'test2'


def test_add_all_new(parent_config, new_config):
    """Test if using ~new adds all new dimensions"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('~new')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_add_all_changed(parent_config, changed_config):
    """Test if using ~changed adds all changed dimensions"""
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)
    branch_builder.add_dimension('~changed')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['add']) == 1


def test_remove_all_missing(parent_config, missing_config):
    """Test if using ~missing removes all missing dimensions"""
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)
    branch_builder.remove_dimension('~missing')

    assert len(branch_builder.conflicts) == 1

    conflict = branch_builder.conflicts[0]

    assert conflict.is_solved is True
    assert len(branch_builder.operations['remove']) == 1


def test_keyword_hit_given_status(parent_config, new_config):
    """Test if using ~new adds only new dimensions"""
    del new_config['metadata']['user_args'][0]
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('~new')

    assert len(branch_builder.conflicts) == 2
    assert len(list(branch_builder.filter_conflicts(lambda c: c.is_solved))) == 1
    assert len(list(branch_builder.filter_conflicts(lambda c: not c.is_solved))) == 1


def test_bad_keyword(parent_config, missing_config):
    """Test that bad keywords are handle silently"""
    branch_builder = ExperimentBranchBuilder(parent_config, missing_config)

    with pytest.raises(ValueError):
        branch_builder.remove_dimension('~test')


def test_good_wildcard(parent_config, new_config):
    """Test if using wildcard adds all dimensions with prefix"""
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('w*')

    assert len(branch_builder.conflicts) == 1
    assert len(list(branch_builder.filter_conflicts(lambda c: c.is_solved))) == 1


def test_wildcard_different_status(parent_config, new_config):
    """Test if using wildcard adds only dimensions with valid status"""
    parent_config['metadata']['user_args'].append('-wow~normal(1,10)')
    branch_builder = ExperimentBranchBuilder(parent_config, new_config)
    branch_builder.add_dimension('w*')

    assert len(branch_builder.conflicts) == 2
    assert len(list(branch_builder.filter_conflicts(lambda c: c.is_solved))) == 1


def test_old_dimension_value(parent_config, changed_config):
    """Test if experiment is not corrupted and old value is the same"""
    branch_builder = ExperimentBranchBuilder(parent_config, changed_config)

    assert branch_builder.get_old_dimension_value('/y') is not None


def test_commandline_solving(parent_config, cl_config):
    """Test if all markers work for conflicts solving through the command line"""
    cl_config['metadata']['user_args'][2] = '-w_d~normal(0,1)'

    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)

    assert len(branch_builder.conflicting_space) == 2
    assert len(branch_builder.conflicts) == 4
    assert len(list(branch_builder.filter_conflicts(filter_true))) == 4
    assert len(branch_builder.operations['add']) == 1
    assert len(branch_builder.operations['rename']) == 1
    assert len(branch_builder.operations['remove']) == 1


def test_adapter_add_new(parent_config, cl_config):
    """Test if a DimensionAddition is created when solving a new conflict"""
    indexes = [3, 1, 0]
    for i in indexes:
        del cl_config['metadata']['user_args'][i]

    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)
    adapters = branch_builder.create_adapters().adapters

    assert len(adapters) == 1
    assert isinstance(adapters[0], Adapters.DimensionAddition)


def test_adapter_add_changed(parent_config, cl_config):
    """Test if a DimensionPriorChange is created when solving a new conflict"""
    indexes = [3, 2, 0]
    for i in indexes:
        del cl_config['metadata']['user_args'][i]

    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)
    adapters = branch_builder.create_adapters().adapters

    assert len(adapters) == 1
    assert isinstance(adapters[0], Adapters.DimensionPriorChange)


def test_adapter_remove_missing(parent_config, cl_config):
    """Test if a DimensionDeletion is created when solving a new conflict"""
    indexes = [2, 1, 0]
    for i in indexes:
        del cl_config['metadata']['user_args'][i]

    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)
    adapters = branch_builder.create_adapters().adapters

    assert len(adapters) == 1
    assert isinstance(adapters[0], Adapters.DimensionDeletion)


def test_adapter_rename_missing(parent_config, cl_config):
    """Test if a DimensionRenaming is created when solving a new conflict"""
    indexes = [3, 1]
    for i in indexes:
        del cl_config['metadata']['user_args'][i]

    cl_config['metadata']['user_args'][1] = '-w_d~uniform(0,1)'
    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)
    adapters = branch_builder.create_adapters().adapters

    assert len(adapters) == 1
    assert isinstance(adapters[0], Adapters.DimensionRenaming)


def test_adapter_rename_different_prior(parent_config, cl_config):
    """Test if a DimensionRenaming is created when solving a new conflict"""
    indexes = [3, 1]
    for i in indexes:
        del cl_config['metadata']['user_args'][i]

    cl_config['metadata']['user_args'][1] = '-w_d~normal(0,1)'

    branch_builder = ExperimentBranchBuilder(parent_config, cl_config)
    adapters = branch_builder.create_adapters().adapters

    assert len(adapters) == 1
    assert isinstance(adapters[0], Adapters.CompositeAdapter)

    adapters = adapters[0].adapters

    assert len(adapters) == 2
    assert isinstance(adapters[0], Adapters.DimensionRenaming)
    assert isinstance(adapters[1], Adapters.DimensionPriorChange)
