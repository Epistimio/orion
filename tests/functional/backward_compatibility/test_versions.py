#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests to verify backward compatibility."""
import os
import shutil
import subprocess

from pymongo import MongoClient
import pytest


DIRNAME = os.path.dirname(os.path.abspath(__file__))

SCRIPT_PATH = os.path.join(DIRNAME, 'black_box.py')
CONFIG_FILE = os.path.join(DIRNAME, 'random.yaml')


def get_package(version):
    """Get package name based on version.

    Package changed to orion instead of orion.core at 0.1.6
    """
    if version >= '0.1.6':
        return 'orion'

    return 'orion.core'


# Ignore pre-0.1.3 because there was no PickleDB backend.
VERSIONS = ['0.1.3', '0.1.4', '0.1.5', '0.1.6']


def clean_mongodb():
    """Clean collections."""
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    database.experiments.drop()
    database.lying_trials.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()
    client.close()


def set_env():
    """Set the env vars for the db.

    Notes
    -----
    This function expects the env var ORION_DB_TYPE to be set.

    """
    db_type = os.environ['ORION_DB_TYPE']
    if db_type == 'pickleddb':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save.pkl')
        os.environ['ORION_DB_ADDRESS'] = file_path
        if os.path.exists(file_path):
            os.remove(file_path)
    elif db_type == 'mongodb':
        os.environ['ORION_DB_ADDRESS'] = 'mongodb://user:pass@localhost'
        os.environ['ORION_DB_NAME'] = 'orion_test'
        clean_mongodb()


def execute(command):
    """Execute a command and return the decoded stdout."""
    out = subprocess.check_output(command.split())
    return out.decode('utf-8')


def setup_virtualenv(version):
    """Create a virtualenv and install Oríon in it for the given version"""
    virtualenv_dir = get_virtualenv_dir(version)
    if os.path.exists(virtualenv_dir):
        shutil.rmtree(virtualenv_dir)
    execute('virtualenv {}'.format(virtualenv_dir))
    command = '{}/bin/pip install --ignore-installed {}=={}'.format(
        virtualenv_dir, get_package(version), version)
    execute(command)


def get_version(orion_script):
    """Get Oríon version for given Oríon executer's path."""
    command = '{} --version'.format(orion_script)
    stdout = subprocess.check_output(command, shell=True)

    return stdout.decode('utf-8').strip('\n')


def get_virtualenv_dir(version):
    """Get virtualenv directory for given version."""
    return 'version-{}'.format(version)


@pytest.fixture(scope='class', params=VERSIONS)
def fill_db(request):
    """Add experiments and trials in DB for given version of Oríon."""
    set_env()

    version = request.param

    setup_virtualenv(version)

    orion_script = os.path.join(get_virtualenv_dir(version), 'bin', 'orion')

    orion_version = get_version(orion_script)
    assert orion_version == 'orion {}'.format(version)

    print(execute(' '.join([
        orion_script, '-vv', 'init_only', '--name', 'init',
        '--config', CONFIG_FILE,
        SCRIPT_PATH, '-x~uniform(-50,50)'])))

    print(execute(' '.join([
        orion_script, '-vv', 'init_only', '--name', 'init',
        '--branch', 'init-branch-old',
        '--config', CONFIG_FILE])))

    print(execute(' '.join([
        orion_script, '-vv', 'hunt', '--name', 'hunt',
        '--config', CONFIG_FILE,
        SCRIPT_PATH, '-x~uniform(-50,50)'])))

    print(execute(' '.join([
        orion_script, '-vv', 'hunt', '--name', 'hunt',
        '--branch', 'hunt-branch-old',
        '--config', CONFIG_FILE])))

    orion_version = get_version('orion')
    assert orion_version != 'orion {}'.format(version)


@pytest.mark.usefixtures('fill_db')
class TestBackwardCompatibility:
    """Tests for backward compatibility between Oríon versions."""

    def test_db(self):
        """Verify test-db command"""
        out = execute('orion test-db')
        assert 'Failure' not in out

    def test_list(self):
        """Verify list command"""
        out = execute('orion list')
        assert 'init-v1' in out
        assert 'init-branch-old-v1' in out
        assert 'hunt-v1' in out
        assert 'hunt-branch-old-v1' in out

    def test_status(self):
        """Verify status command"""
        out = execute('orion status')
        assert 'init-v1' in out
        assert 'init-branch-old-v1' in out
        assert 'hunt-v1' in out
        assert 'hunt-branch-old-v1' in out

    def test_info(self):
        """Verify info command"""
        out = execute('orion info --name hunt')
        assert 'name: hunt' in out

    def test_init_only(self):
        """Verify init_only command"""
        print(execute(' '.join([
            'orion', 'init_only', '--name', 'init',
            '--branch', 'init-branch'])))

    def test_hunt(self):
        """Verify hunt command"""
        print(execute(' '.join([
            'orion', 'hunt', '--name', 'hunt',
            '--branch', 'hunt-branch'])))

    # orion.core.cli.main('init-only') # TODO: deprecate init_only
