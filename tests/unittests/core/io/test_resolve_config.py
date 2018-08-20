#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.resolve_config`."""

import hashlib
import os
import shutil
import socket

import git
import pytest

import orion.core.io.resolve_config as resolve_config


@pytest.fixture
def force_is_exe(monkeypatch):
    """Mock resolve_config to recognize any string as an executable script."""
    def is_exe(path):
        return True

    monkeypatch.setattr(resolve_config, "is_exe", is_exe)


def test_fetch_default_options():
    """Verify default options"""
    resolve_config.DEF_CONFIG_FILES_PATHS = []
    default_config = resolve_config.fetch_default_options()

    assert default_config['algorithms'] == 'random'
    assert default_config['database']['host'] == socket.gethostbyname(socket.gethostname())
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


@pytest.mark.usefixtures("version_XYZ")
def test_fetch_metadata_orion_version():
    """Verify orion version"""
    metadata = resolve_config.fetch_metadata({})
    assert metadata['orion_version'] == 'XYZ'


def test_fetch_metadata_executable_users_script(script_path):
    """Verify executable user script with absolute path"""
    cmdargs = {'user_args': [script_path]}
    metadata = resolve_config.fetch_metadata(cmdargs)
    assert metadata['user_script'] == os.path.abspath(script_path)


def test_fetch_metadata_non_executable_users_script():
    """Verify executable user script keeps given path"""
    cmdargs = {'user_args': ['tests/functional/demo/orion_config.yaml']}
    metadata = resolve_config.fetch_metadata(cmdargs)
    assert metadata['user_script'] == 'tests/functional/demo/orion_config.yaml'


def test_fetch_metadata_not_existed_path():
    """Verfiy the raise of error when user_script path does not exist"""
    path = 'dummy/path'
    cmdargs = {'user_args': [path]}
    with pytest.raises(OSError) as exc_info:
        resolve_config.fetch_metadata(cmdargs)
    assert "The path specified for the script does not exist" in str(exc_info.value)


@pytest.mark.usefixtures()
def test_fetch_metadata_user_args(script_path):
    """Verify user args"""
    user_args = [os.path.abspath(script_path)] + list(map(str, range(10)))
    cmdargs = {'user_args': user_args}
    metadata = resolve_config.fetch_metadata(cmdargs)
    assert metadata['user_script'] == user_args[0]
    assert metadata['user_args'] == user_args[1:]


@pytest.mark.usefixtures("with_user_tsirif")
def test_fetch_metadata_user_tsirif():
    """Verify user name"""
    metadata = resolve_config.fetch_metadata({})
    assert metadata['user'] == "tsirif"


def test_fetch_metadata():
    """Verify no additional data is stored in metadata"""
    metadata = resolve_config.fetch_metadata({})
    len(metadata) == 4


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


@pytest.fixture
def repo():
    """Create a dummy repo for the tests."""
    os.chdir('../')
    os.makedirs('dummy_orion')
    os.chdir('dummy_orion')
    repo = git.Repo.init('.')
    with open('README.md', 'w+') as f:
        f.write('dummy content')
    repo.git.add('README.md')
    repo.index.commit('initial commit')
    repo.create_head('master')
    repo.git.checkout('master')
    yield repo
    os.chdir('../')
    shutil.rmtree('dummy_orion')
    os.chdir('orion')


def test_infer_versioning_metadata_on_clean_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the user's repo is clean:
    `is_dirty`, `active_branch` and `diff_sha`.
    """
    vcs = resolve_config.infer_versioning_metadata('.git')
    assert not vcs['is_dirty']
    assert vcs['active_branch'] == 'master'
    # the diff should be empty so the diff_sha should be equal to the diff sha of an empty string
    assert vcs['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()


def test_infer_versioning_metadata_on_dirty_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the uers's repo is dirty:
    `is_dirty`, `HEAD_sha`, `active_branch` and `diff_sha`.
    """
    existing_metadata = {}
    existing_metadata['user_script'] = '.git'
    vcs = resolve_config.infer_versioning_metadata('.git')
    repo.create_head('feature')
    repo.git.checkout('feature')
    with open('README.md', 'w+') as f:
        f.write('dummy dummy content')
    vcs = resolve_config.infer_versioning_metadata('.git')
    assert vcs['is_dirty']
    assert vcs['active_branch'] == 'feature'
    assert vcs['diff_sha'] != hashlib.sha256(''.encode('utf-8')).hexdigest()
    repo.git.add('README.md')
    commit = repo.index.commit('Added dummy_file')
    vcs = resolve_config.infer_versioning_metadata('.git')
    assert not vcs['is_dirty']
    assert vcs['HEAD_sha'] == commit.hexsha
    assert vcs['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()


def test_fetch_user_repo_on_non_repo():
    """
    Test if `fetch_user_repo` raises a warning when user's script
    is not a git repo
    """
    with pytest.raises(RuntimeError) as exc_info:
        resolve_config.fetch_user_repo('.')
    assert "Script {} should be in a git repository".format(os.getcwd()) in str(exc_info.value)


def test_infer_versioning_metadata_on_detached_head(repo):
    """Test in the case of a detached head."""
    with open('README.md', 'w+') as f:
        f.write('dummy contentt')
    repo.git.add('README.md')
    repo.index.commit('2nd commit')
    existing_metadata = {}
    existing_metadata['user_script'] = '.git'
    repo.head.reference = repo.commit('HEAD~1')
    assert repo.head.is_detached
    vcs = resolve_config.infer_versioning_metadata('.git')
    assert vcs['active_branch'] is None
