#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.resolve_config`."""

import hashlib
import os
import shutil

import git

from orion.core.cli import resolve_config

join = os.path.join


def test_infer_versioning_metadata():
    """
    Test how `infer_versioning_metadata` fills its different fields :
    `is_dirty`, `HEAD_sha`, `active_branch` and `diff_sha`.
    """
    os.chdir('../')
    if not os.path.exists('dummy_orion'):
        os.makedirs('dummy_orion')
        os.chdir('dummy_orion')
        git.Repo.clone_from('https://github.com/ReyhaneAskari/dummy_orion.git', '.')

    test_repo = git.Repo('.git')

    existing_metadata = {}
    existing_metadata['user_script'] = '.git'
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert not existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['HEAD_sha'] == 'd316303cfcd1394df861ec33658faf42083d3d55'
    assert existing_metadata['VCS']['active_branch'] == 'master'
    # the diff should be empty so the diff_sha should be equal to the diff sha of an empty string
    assert existing_metadata['VCS']['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()

    test_repo.create_head('feature')
    test_repo.git.checkout('feature')
    with open('README.md', 'w+') as f:
        f.write('dummy content')
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['active_branch'] == 'feature'
    assert existing_metadata['VCS']['diff_sha'] != hashlib.sha256(''.encode('utf-8')).hexdigest()
    test_repo.git.add('README.md')
    commit = test_repo.index.commit('Added dummy_file')
    existing_metadata = resolve_config.infer_versioning_metadata(existing_metadata)
    assert not existing_metadata['VCS']['is_dirty']
    assert existing_metadata['VCS']['HEAD_sha'] == commit.hexsha
    assert existing_metadata['VCS']['diff_sha'] == hashlib.sha256(''.encode('utf-8')).hexdigest()

    os.chdir('../')
    shutil.rmtree('dummy_orion')
