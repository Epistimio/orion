#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.resolve_config`."""


import os

import orion.core.io.resolve_config as resolve_config


def test_fetch_default_options():
    """Verify default options"""
    default_config = resolve_config.fetch_default_options()

    assert default_config['algorithms'] == 'random'
    assert default_config['database']['host'] == '127.0.1.1'
    assert default_config['database']['name'] == 'orion'
    assert default_config['database']['type'] == 'MongoDB'

    assert default_config['max_trials'] == float('inf')
    assert default_config['name'] is None
    assert default_config['pool_size'] == 10


def test_fetch_env_vars():
    """Verify env vars are fetched properly"""
    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {'database': {}}

    db_name = "orion_test"

    os.environ['ORION_DB_NAME'] = db_name

    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {'database': {'name': 'orion_test'}}

    db_type = "MongoDB"
    os.environ['ORION_DB_TYPE'] = db_type

    env_vars_config = resolve_config.fetch_env_vars()
    assert env_vars_config == {'database': {'name': db_name, 'type': db_type}}


def test_fetch_config_no_hit():
    """Verify fetch_config returns empty dict on no config file path"""
    config = resolve_config.fetch_config({"config": ""})
    assert config == {}


def test_fetch_config(config_file):
    """Verify fetch_config returns valid dictionnary"""
    config = resolve_config.fetch_config({"config": config_file})

    assert config['algorithms'] == 'random'
    assert config['database']['host'] == 'mongodb://user:pass@localhost'
    assert config['database']['name'] == 'orion_test'
    assert config['database']['type'] == 'mongodb'

    assert config['max_trials'] == 100
    assert config['name'] == 'voila_voici'
    assert config['pool_size'] == 1


def test_merge_configs_update_two():
    """Ensure update on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'a': 3}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 3, 'b': 2}


def test_merge_configs_update_three():
    """Ensure two updates on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'a': 3}
    c = {'b': 4}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {'a': 3, 'b': 4}


def test_merge_configs_update_four():
    """Ensure three updates on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'a': 3}
    c = {'b': 4}
    d = {'a': 5, 'b': 6}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {'a': 5, 'b': 6}


def test_merge_configs_extend_two():
    """Ensure extension on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'c': 3}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': 2, 'c': 3}


def test_merge_configs_extend_three():
    """Ensure two extensions on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'c': 3}
    c = {'d': 4}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_merge_configs_extend_four():
    """Ensure three extensions on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'c': 3}
    c = {'d': 4}
    d = {'e': 5}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}


def test_merge_configs_update_extend_two():
    """Ensure update and extension on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'b': 3, 'c': 4}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': 3, 'c': 4}


def test_merge_configs_update_extend_three():
    """Ensure two updates and extensions on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'b': 3, 'c': 4}
    c = {'a': 5, 'd': 6}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {'a': 5, 'b': 3, 'c': 4, 'd': 6}


def test_merge_configs_update_extend_four():
    """Ensure three updates and extensions on first level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'b': 3, 'c': 4}
    c = {'a': 5, 'd': 6}
    d = {'d': 7, 'e': 8}

    m = resolve_config.merge_configs(a, b, c, d)

    assert m == {'a': 5, 'b': 3, 'c': 4, 'd': 7, 'e': 8}


def test_merge_sub_configs_update_two():
    """Ensure updating to second level is fine"""
    a = {'a': 1, 'b': 2}
    b = {'b': {'c': 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': {'c': 3}}


def test_merge_sub_configs_sub_update_two():
    """Ensure updating on second level is fine"""
    a = {'a': 1, 'b': {'c': 2}}
    b = {'b': {'c': 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': {'c': 3}}

    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'b': {'c': 4}}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': {'c': 4, 'd': 3}}


def test_merge_sub_configs_sub_extend_two():
    """Ensure updating to third level from second level is fine"""
    a = {'a': 1, 'b': {'c': 2}}
    b = {'d': {'e': 3}}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': {'c': 2}, 'd': {'e': 3}}

    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'b': {'e': {'f': 4}}}

    m = resolve_config.merge_configs(a, b)

    assert m == {'a': 1, 'b': {'c': 2, 'd': 3, 'e': {'f': 4}}}


def test_merge_sub_configs_update_three():
    """Ensure updating twice to third level from second level is fine"""
    a = {'a': 1, 'b': {'c': 2}}
    b = {'b': {'c': 3}}
    c = {'b': {'c': {'d': 4}}}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {'a': 1, 'b': {'c': {'d': 4}}}

    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'b': {'c': 4}}
    c = {'b': {'c': {'e': 5}}}

    m = resolve_config.merge_configs(a, b, c)

    assert m == {'a': 1, 'b': {'c': {'e': 5}, 'd': 3}}


def test_infer_versioning_metadata():
    """Verify infer_versioning_metadata does nothing so far

    Test should be broken once the function is implemented
    """
    metadata = {'hello': {'world': 0}}
    assert resolve_config.infer_versioning_metadata(metadata) == metadata
