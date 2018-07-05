#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.resolve_config`."""

import hashlib
import os
import shutil

import git
import pytest

from orion.core.cli import resolve_config

join = os.path.join


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


def test_infer_versioning_metadata_on_clean_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the user's repo is clean:
    `is_dirty`, `active_branch` and `diff_sha`.
    """
    existing_metadata = {}
    existing_metadata['user_script'] = '.git'
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert not existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['active_branch'] == 'master'
    # the diff should be empty so the diff_sha should be equal to the diff sha of an empty string
    assert existing_metadata['VCS']['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()


def test_infer_versioning_metadata_on_dirty_repo(repo):
    """
    Test how `infer_versioning_metadata` fills its different fields
    when the uers's repo is dirty:
    `is_dirty`, `HEAD_sha`, `active_branch` and `diff_sha`.
    """
    existing_metadata = {}
    existing_metadata['user_script'] = '.git'
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    repo.create_head('feature')
    repo.git.checkout('feature')
    with open('README.md', 'w+') as f:
        f.write('dummy dummy content')
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['active_branch'] == 'feature'
    assert existing_metadata['VCS']['diff_sha'] != hashlib.sha256(''.encode('utf-8')).hexdigest()
    repo.git.add('README.md')
    commit = repo.index.commit('Added dummy_file')
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert not existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['HEAD_sha'] == commit.hexsha
    assert existing_metadata['VCS']['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()


def test_fetch_user_repo_on_non_repo():
    """
    Test if `fetch_user_repo` raises a warning when user's script
    is not a git repo
    """
    with pytest.warns(UserWarning):
        resolve_config.fetch_user_repo('.')
