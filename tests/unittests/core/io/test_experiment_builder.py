#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.experiment_builder`."""

import pytest

from orion.core.io import resolve_config
from orion.core.io.experiment_builder import ExperimentBuilder


@pytest.mark.usefixtures("clean_db")
def test_fetch_local_config(config_file):
    """Test local config (default, env_vars, cmdconfig, cmdargs)"""
    cmdargs = {"config": config_file}
    local_config = ExperimentBuilder().fetch_full_config(cmdargs, use_db=False)

    assert local_config['algorithms'] == 'random'
    assert local_config['database']['host'] == 'mongodb://user:pass@localhost'
    assert local_config['database']['name'] == 'orion_test'
    assert local_config['database']['type'] == 'mongodb'
    assert local_config['max_trials'] == 100
    assert local_config['name'] == 'voila_voici'
    assert local_config['pool_size'] == 1


@pytest.mark.usefixtures("clean_db")
def test_fetch_local_config_from_incomplete_config(incomplete_config_file):
    """Test local config with incomplete user configuration file
    (default, env_vars, cmdconfig, cmdargs)

    This is to ensure merge_configs update properly the subconfigs
    """
    resolve_config.DEF_CONFIG_FILES_PATHS = []
    cmdargs = {"config": incomplete_config_file}
    local_config = ExperimentBuilder().fetch_full_config(cmdargs, use_db=False)

    assert local_config['algorithms'] == 'random'
    assert local_config['database']['host'] == 'mongodb://user:pass@localhost'
    assert local_config['database']['name'] == 'orion'
    assert local_config['database']['type'] == 'incomplete'
    assert local_config['max_trials'] == float('inf')
    assert local_config['name'] == 'incomplete'
    assert local_config['pool_size'] == 10


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_fetch_config_from_db_no_hit(config_file, random_dt):
    """Verify that fetch_config_from_db returns an empty dict when the experiment is not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}
    db_config = ExperimentBuilder().fetch_config_from_db(cmdargs)
    assert db_config == {}


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_fetch_config_from_db_hit(config_file, exp_config, random_dt):
    """Verify db config when experiment is in db"""
    cmdargs = {'name': 'supernaedo2', 'config': config_file}
    db_config = ExperimentBuilder().fetch_config_from_db(cmdargs)

    assert db_config['name'] == exp_config[0][0]['name']
    assert db_config['refers'] == exp_config[0][0]['refers']
    assert db_config['metadata'] == exp_config[0][0]['metadata']
    assert db_config['pool_size'] == exp_config[0][0]['pool_size']
    assert db_config['max_trials'] == exp_config[0][0]['max_trials']
    assert db_config['algorithms'] == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif",
                         "mock_infer_versioning_metadata")
def test_fetch_full_config_new_config(config_file, exp_config, random_dt, script_path):
    """Verify full config with new config (causing branch)"""
    cmdargs = {'name': 'supernaedo2',
               'config': config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}
    full_config = ExperimentBuilder().fetch_full_config(cmdargs)
    cmdconfig = ExperimentBuilder().fetch_file_config(cmdargs)

    full_config['metadata']['orion_version'] = exp_config[0][0]['metadata']['orion_version']

    assert full_config['name'] == exp_config[0][0]['name']
    assert full_config['refers'] == exp_config[0][0]['refers']
    assert full_config['metadata'] == exp_config[0][0]['metadata']
    assert full_config['pool_size'] == cmdconfig['pool_size']
    assert full_config['max_trials'] == cmdconfig['max_trials']
    assert full_config['algorithms'] == cmdconfig['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif",
                         "mock_infer_versioning_metadata")
def test_fetch_full_config_old_config(old_config_file, exp_config, random_dt, script_path):
    """Verify full config with old config (not causing branch)"""
    cmdargs = {'name': 'supernaedo2',
               'config': old_config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}

    full_config = ExperimentBuilder().fetch_full_config(cmdargs)
    cmdconfig = ExperimentBuilder().fetch_file_config(cmdargs)

    full_config['metadata']['orion_version'] = exp_config[0][0]['metadata']['orion_version']

    assert full_config['name'] == exp_config[0][0]['name']
    assert full_config['refers'] == exp_config[0][0]['refers']
    assert full_config['metadata'] == exp_config[0][0]['metadata']
    assert full_config['pool_size'] == cmdconfig['pool_size']
    assert full_config['max_trials'] == cmdconfig['max_trials']
    assert full_config['algorithms'] == cmdconfig['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_fetch_full_config_no_hit(config_file, exp_config, random_dt):
    """Verify full config when experiment not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}
    full_config = ExperimentBuilder().fetch_full_config(cmdargs)

    assert full_config['name'] == 'supernaekei'
    assert full_config['algorithms'] == 'random'
    assert full_config['max_trials'] == 100
    assert full_config['name'] == 'supernaekei'
    assert full_config['pool_size'] == 1
    assert full_config['metadata']['user'] == 'tsirif'
    assert 'datetime' not in full_config['metadata']
    assert 'refers' not in full_config


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_view_from_no_hit(config_file, create_db_instance, exp_config):
    """Try building experiment view when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file}

    with pytest.raises(ValueError) as exc_info:
        ExperimentBuilder().build_view_from(cmdargs)
    assert "No experiment with given name 'supernaekei' for user 'tsirif'" in str(exc_info.value)


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_view_from(config_file, create_db_instance, exp_config, random_dt):
    """Try building experiment view when in db"""
    cmdargs = {'name': 'supernaedo2', 'config': config_file}
    exp_view = ExperimentBuilder().build_view_from(cmdargs)

    assert exp_view._experiment._init_done is True
    assert exp_view._experiment._db._database is create_db_instance
    assert exp_view._id == exp_config[0][0]['_id']
    assert exp_view.name == exp_config[0][0]['name']
    assert exp_view.configuration['refers'] == exp_config[0][0]['refers']
    assert exp_view.metadata == exp_config[0][0]['metadata']
    assert exp_view._experiment._last_fetched == exp_config[0][0]['metadata']['datetime']
    assert exp_view.pool_size == exp_config[0][0]['pool_size']
    assert exp_view.max_trials == exp_config[0][0]['max_trials']
    assert exp_view.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_from_no_hit(config_file, create_db_instance, exp_config, random_dt, script_path):
    """Try building experiment when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file,
               'user_args': [script_path,
                             'x~uniform(0,10)']}

    with pytest.raises(ValueError) as exc_info:
        ExperimentBuilder().build_view_from(cmdargs)
    assert "No experiment with given name 'supernaekei' for user 'tsirif'" in str(exc_info.value)

    exp = ExperimentBuilder().build_from(cmdargs)

    assert exp._init_done is True
    assert exp._db is create_db_instance
    assert exp.name == cmdargs['name']
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'tsirif'
    assert exp.metadata['user_script'] == cmdargs['user_args'][0]
    assert exp.metadata['user_args'] == cmdargs['user_args'][1:]
    assert exp._last_fetched == random_dt
    assert exp.pool_size == 1
    assert exp.max_trials == 100
    assert exp.algorithms.configuration == {'random': {}}


@pytest.mark.usefixtures("version_XYZ", "clean_db", "null_db_instances", "with_user_tsirif",
                         "mock_infer_versioning_metadata")
def test_build_from_hit(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment when in db (no branch)"""
    cmdargs = {'name': 'supernaedo2',
               'config': old_config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}

    # Test that experiment already exists
    ExperimentBuilder().build_view_from(cmdargs)
    exp = ExperimentBuilder().build_from(cmdargs)

    assert exp._init_done is True
    assert exp._db is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp._last_fetched == exp_config[0][0]['metadata']['datetime']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']


@pytest.mark.usefixtures("version_XYZ", "clean_db", "null_db_instances", "with_user_tsirif")
def test_build_from_config_no_hit(config_file, create_db_instance, exp_config, random_dt,
                                  script_path):
    """Try building experiment from config when not in db"""
    cmdargs = {'name': 'supernaekei', 'config': config_file,
               'user_args': [script_path,
                             '-x~uniform(0,10)']}

    with pytest.raises(ValueError) as exc_info:
        ExperimentBuilder().build_view_from(cmdargs)
    assert "No experiment with given name 'supernaekei' for user 'tsirif'" in str(exc_info.value)

    full_config = ExperimentBuilder().fetch_full_config(cmdargs)
    exp = ExperimentBuilder().build_from_config(full_config)

    assert exp._init_done is True
    assert exp._db is create_db_instance
    assert exp.name == cmdargs['name']
    assert exp.configuration['refers'] == {'adapter': [], 'parent_id': None, 'root_id': exp._id}
    assert exp.metadata['datetime'] == random_dt
    assert exp.metadata['user'] == 'tsirif'
    assert exp.metadata['user_script'] == cmdargs['user_args'][0]
    assert exp.metadata['user_args'] == cmdargs['user_args'][1:]
    assert exp._last_fetched == random_dt
    assert exp.pool_size == 1
    assert exp.max_trials == 100
    assert not exp.is_done
    assert exp.algorithms.configuration == {'random': {}}


@pytest.mark.usefixtures("clean_db", "null_db_instances", "with_user_tsirif")
def test_build_from_config_hit(old_config_file, create_db_instance, exp_config, script_path):
    """Try building experiment from config when in db (no branch)"""
    cmdargs = {'name': 'supernaedo2',
               'config': old_config_file,
               'user_args': [script_path,
                             "--encoding_layer~choices(['rnn', 'lstm', 'gru'])",
                             "--decoding_layer~choices(['rnn', 'lstm_with_attention', 'gru'])"]}

    # Test that experiment already exists
    ExperimentBuilder().build_view_from(cmdargs)

    exp_view = ExperimentBuilder().build_view_from(cmdargs)
    exp = ExperimentBuilder().build_from_config(exp_view.configuration)

    assert exp._init_done is True
    assert exp._db is create_db_instance
    assert exp._id == exp_config[0][0]['_id']
    assert exp.name == exp_config[0][0]['name']
    assert exp.configuration['refers'] == exp_config[0][0]['refers']
    assert exp.metadata == exp_config[0][0]['metadata']
    assert exp._last_fetched == exp_config[0][0]['metadata']['datetime']
    assert exp.pool_size == exp_config[0][0]['pool_size']
    assert exp.max_trials == exp_config[0][0]['max_trials']
    assert exp.algorithms.configuration == exp_config[0][0]['algorithms']
