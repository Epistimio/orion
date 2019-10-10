#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.experiment_builder`."""
import pytest

import orion.core.io.experiment_builder as experiment_builder
from orion.core.io.experiment_builder import (
    build_from_args, build_view_from_args, get_cmd_config,
    setup_storage)
from orion.core.utils.exceptions import NoConfigurationError


def get_db(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db


def get_view_db(exp):
    """Transitional method to move away from mongodb"""
    return exp._experiment._storage._storage._db


@pytest.fixture
def init_storage(clean_db, test_config):
    """Create the storage singleton."""
    setup_storage(
        storage={
            'type': 'legacy',
            'config': {
                'database': {
                    'type': 'mongodb',
                    'name': 'orion_test',
                    'host': 'mongodb://user:pass@localhost'}}})


def test_get_cmd_config(config_file):
    """Test local config (cmdconfig, cmdargs)"""
    cmdargs = {"config": config_file}
    local_config = experiment_builder.get_cmd_config(cmdargs)

    assert local_config['algorithms'] == 'random'
    assert local_config['producer'] == {'strategy': 'NoParallelStrategy'}
    assert local_config['database']['host'] == 'mongodb://user:pass@localhost'
    assert local_config['database']['name'] == 'orion_test'
    assert local_config['database']['type'] == 'mongodb'
    assert local_config['max_trials'] == 100
    assert local_config['name'] == 'voila_voici'
    assert local_config['pool_size'] == 1


def test_get_cmd_config_from_incomplete_config(incomplete_config_file):
    """Test local config with incomplete user configuration file
    (default, env_vars, cmdconfig, cmdargs)

    This is to ensure merge_configs update properly the subconfigs
    """
    cmdargs = {"config": incomplete_config_file}
    local_config = experiment_builder.get_cmd_config(cmdargs)

    assert 'algorithms' not in local_config
    assert 'name' not in local_config['database']
    assert 'max_trials' not in local_config
    assert 'pool_size' not in local_config
    assert local_config['database']['host'] == 'mongodb://user:pass@localhost'
    assert local_config['database']['type'] == 'incomplete'
    assert local_config['name'] == 'incomplete'


@pytest.mark.usefixtures('init_storage')
def test_fetch_config_from_db_no_hit(config_file, random_dt):
    """Verify that fetch_config_from_db returns an empty dict when the experiment is not in db"""
    db_config = experiment_builder.fetch_config_from_db(name='supernaekei')
    assert db_config == {}


@pytest.mark.usefixtures('with_user_tsirif', 'init_storage')
def test_fetch_config_from_db_hit(exp_config):
    """Verify db config when experiment is in db"""
    db_config = experiment_builder.fetch_config_from_db(name='supernaedo2-dendi')

    assert db_config['name'] == exp_config[0][0]['name']
    assert db_config['refers'] == exp_config[0][0]['refers']
    assert db_config['metadata'] == exp_config[0][0]['metadata']
    assert db_config['pool_size'] == exp_config[0][0]['pool_size']
    assert db_config['max_trials'] == exp_config[0][0]['max_trials']
    assert db_config['algorithms'] == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_view_from_no_hit(config_file, create_db_instance, exp_config):
    """Try building experiment view when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}

    with pytest.raises(ValueError) as exc_info:
        build_view_from_args(cmdargs)
    assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_view_from(config_file, create_db_instance, exp_config, random_dt):
    """Try building experiment view when in db"""
    cmdargs = {'name': 'supernaedo2-dendi', 'config': config_file}
    exp_view = build_view_from_args(cmdargs)

    assert exp_view._experiment._init_done is False
    assert get_view_db(exp_view) is create_db_instance
    assert exp_view._id == exp_config[0][0]['_id']
    assert exp_view.name == exp_config[0][0]['name']
    assert exp_view.configuration['refers'] == exp_config[0][0]['refers']
    assert exp_view.metadata == exp_config[0][0]['metadata']
    assert exp_view.pool_size == exp_config[0][0]['pool_size']
    assert exp_view.max_trials == exp_config[0][0]['max_trials']
    # TODO: Views are not fully configured until configuration is refactored
    # assert exp_view.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_dendi")
def test_build_from_no_hit(config_file, create_db_instance, exp_config, random_dt, script_path):
    """Try building experiment when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file,
               'user_args': [script_path,
                             'x~uniform(0,10)']}

    with pytest.raises(ValueError) as exc_info:
        build_view_from_args(cmdargs)
    assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)

    exp = build_from_args(cmdargs)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp.name == cmdargs['name']
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'dendi'
    assert exp.metadata['user_script'] == cmdargs['user_args'][0]
    assert exp.metadata['user_args'] == cmdargs['user_args'][1:]
    assert exp.pool_size == 1
    assert exp.max_trials == 100
    assert exp.algorithms.configuration == {'random': {'seed': None}}


@pytest.mark.usefixtures("version_XYZ", "clean_db", "null_db_instances", "with_user_dendi",
                         "mock_infer_versioning_metadata")
def test_build_from_hit(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment when in db (no branch)"""
    cmdargs = {'name': 'supernaedo2-dendi',
               'config': old_config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}

    # Test that experiment already exists
    build_view_from_args(cmdargs)

    exp = build_from_args(cmdargs)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_bouthilx")
def test_build_from_force_user(old_config_file, create_db_instance, exp_config, random_dt):
    """Try building experiment view when in db"""
    cmdargs = {'name': 'supernaedo2', 'config': old_config_file}
    cmdargs['user'] = 'tsirif'
    exp_view = build_from_args(cmdargs)
    assert exp_view.metadata['user'] == 'tsirif'


@pytest.mark.usefixtures("version_XYZ", "clean_db", "null_db_instances", "with_user_tsirif",
                         "mock_infer_versioning_metadata")
def test_build_from_config_no_hit(config_file, create_db_instance, exp_config, random_dt,
                                  script_path):
    """Try building experiment from config when not in db"""
    name = 'supernaekei'
    cmdargs = {'name': name, 'config': config_file,
               'user_args': [script_path,
                             '-x~uniform(0,10)']}

    with pytest.raises(ValueError) as exc_info:
        experiment_builder.build_view(name)
    assert "No experiment with given name 'supernaekei' and version '*'" in str(exc_info.value)

    cmd_config = get_cmd_config(cmdargs)

    exp = experiment_builder.build(space=cmd_config['metadata']['priors'], **cmd_config)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp.name == cmdargs['name']
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'tsirif'
    assert exp.metadata['user_script'] == cmdargs['user_args'][0]
    assert exp.metadata['user_args'] == cmdargs['user_args'][1:]
    assert exp.pool_size == 1
    assert exp.max_trials == 100
    assert not exp.is_done
    assert exp.algorithms.configuration == {'random': {'seed': None}}


@pytest.mark.usefixtures("clean_db", "init_storage")
def test_build_from_config_no_commandline_config():
    """Try building experiment with no commandline configuration."""
    with pytest.raises(NoConfigurationError):
        experiment_builder.build('supernaekei')


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_dendi",
                         "mock_infer_versioning_metadata", "version_XYZ")
def test_build_from_config_hit(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment from config when in db (no branch)"""
    name = 'supernaedo2-dendi'

    cmdargs = {'name': name,
               'config': old_config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}

    # Test that experiment already exists (this should fail otherwise)
    experiment_builder.build_view(name=name)

    config = get_cmd_config(cmdargs)

    exp = experiment_builder.build(space=config['metadata']['priors'], **config)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_dendi")
def test_build_without_config_hit(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment without commandline config when in db (no branch)"""
    name = 'supernaedo2-dendi'

    # Test that experiment already exists (this should fail otherwise)
    experiment_builder.build_view(name=name)

    exp = experiment_builder.build(name=name)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_dendi", "version_XYZ")
def test_build_without_cmd(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment without commandline when in db (no branch)"""
    name = 'supernaedo2-dendi'

    cmdargs = {'name': name,
               'config': old_config_file}

    # Test that experiment already exists (this should fail otherwise)
    build_view_from_args(cmdargs)

    exp = build_from_args(cmdargs)

    assert exp._init_done is True
    assert get_db(exp) is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']
