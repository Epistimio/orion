#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests to verify backward compatibility."""
import os
import shutil
import subprocess

from pymongo import MongoClient
import pytest

from orion.client import create_experiment
from orion.core.io.database import Database, OutdatedDatabaseError
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
import orion.core.io.experiment_builder as experiment_builder
from orion.storage.base import get_storage, Storage
from orion.storage.legacy import Legacy


DIRNAME = os.path.dirname(os.path.abspath(__file__))

PYTHON_SCRIPT_PATH = os.path.join(DIRNAME, 'python_api.py')
SCRIPT_PATH = os.path.join(DIRNAME, 'black_box.py')
CONFIG_FILE = os.path.join(DIRNAME, 'random.yaml')

# Ignore pre-0.1.3 because there was no PickleDB backend.
VERSIONS = ['0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7']


def get_package(version):
    """Get package name based on version.

    Package changed to orion instead of orion.core at 0.1.6
    """
    if version >= '0.1.6':
        return 'orion'

    return 'orion.core'


def get_branch_argument(version):
    """Get argument to branch.

    Before v0.1.8 it was --branch. From v0.1.8 and forward it is now --branch-to.
    """
    return '--branch' if version < '0.1.8' else '--branch-to'


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


def fill_from_cmdline_api(orion_script, version):
    """Add experiments and trials using the commandline API"""
    print(execute(' '.join([
        orion_script, '-vv', 'init_only', '--name', 'init-cmdline',
        '--config', CONFIG_FILE,
        SCRIPT_PATH, '-x~uniform(-50,50)'])))

    print(execute(' '.join([
        orion_script, '-vv', 'init_only', '--name', 'init-cmdline',
        get_branch_argument(version), 'init-cmdline-branch-old',
        '--config', CONFIG_FILE])))

    print(execute(' '.join([
        orion_script, '-vv', 'hunt', '--name', 'hunt-cmdline',
        '--config', CONFIG_FILE,
        SCRIPT_PATH, '-x~uniform(-50,50)'])))

    print(execute(' '.join([
        orion_script, '-vv', 'hunt', '--name', 'hunt-cmdline',
        get_branch_argument(version), 'hunt-cmdline-branch-old',
        '--config', CONFIG_FILE])))


def fill_from_python_api(python_script, version):
    """Add experiments and trials using the python API"""
    print(execute(' '.join([
        python_script, PYTHON_SCRIPT_PATH, version])))


@pytest.fixture(scope='class', params=VERSIONS)
def fill_db(request):
    """Add experiments and trials in DB for given version of Oríon."""
    set_env()

    version = request.param

    setup_virtualenv(version)

    orion_script = os.path.join(get_virtualenv_dir(version), 'bin', 'orion')
    python_script = os.path.join(get_virtualenv_dir(version), 'bin', 'python')

    orion_version = get_version(orion_script)
    assert orion_version == 'orion {}'.format(version)

    fill_from_cmdline_api(orion_script, version)
    if version > '0.1.7':
        fill_from_python_api(python_script, version)

    orion_version = get_version('orion')
    assert orion_version != 'orion {}'.format(version)

    # Should fail before upgrade
    if version < '0.1.6':
        with pytest.raises(OutdatedDatabaseError):
            build_storage()

    print(execute('orion -vv db upgrade -f'))

    return version


def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    Storage.instance = None
    Legacy.instance = None
    Database.instance = None
    MongoDB.instance = None
    PickledDB.instance = None


def build_storage():
    """Build storage from scratch"""
    null_db_instances()
    experiment_builder.setup_storage()

    return get_storage()


@pytest.mark.usefixtures('fill_db')
class TestBackwardCompatibility:
    """Tests for backward compatibility between Oríon versions."""

    def test_db_upgrade(self):
        """Verify db upgrade was successful"""
        storage = build_storage()

        index_info = storage._db.index_information('experiments')
        assert (set([key for key, is_unique in index_info.items() if is_unique]) ==
                set(['_id_', 'name_1_version_1']))

        experiments = storage.fetch_experiments({})
        assert 'version' in experiments[0]
        assert 'priors' in experiments[0]['metadata']

    def test_db_test(self):
        """Verify db test command"""
        out = execute('orion db test')
        assert 'Failure' not in out

    def test_list(self, fill_db):
        """Verify list command"""
        out = execute('orion list')
        assert 'init-cmdline-v1' in out
        assert 'init-cmdline-branch-old-v1' in out
        assert 'hunt-cmdline-v1' in out
        assert 'hunt-cmdline-branch-old-v1' in out

        version = fill_db
        if version > '0.1.7':
            assert 'hunt-python-v1' in out

    def test_status(self, fill_db):
        """Verify status command"""
        out = execute('orion status')
        assert 'init-cmdline-v1' in out
        assert 'init-cmdline-branch-old-v1' in out
        assert 'hunt-cmdline-v1' in out
        assert 'hunt-cmdline-branch-old-v1' in out

        version = fill_db
        if version > '0.1.7':
            assert 'hunt-python-v1' in out

    def test_info_cmdline_api(self):
        """Verify info command from commandline api"""
        out = execute('orion info --name hunt-cmdline')
        assert 'name: hunt-cmdline' in out

    def test_info_python_api(self, fill_db):
        """Verify info command from python api"""
        version = fill_db
        if version < '0.1.8':
            pytest.skip("Python API not supported by {}".format(version))

        out = execute('orion info --name hunt-python')
        assert 'name: hunt-python' in out

    def test_init_only(self):
        """Verify init_only command"""
        print(execute(' '.join([
            'orion', 'init_only', '--name', 'init-cmdline',
            '--branch-to', 'init-cmdline-branch'])))

    def test_hunt_cmdline_api(self):
        """Verify hunt command from cmdline api parent"""
        print(execute(' '.join([
            'orion', 'hunt', '--name', 'hunt-cmdline',
            '--branch-to', 'hunt-cmdline-branch'])))

    def test_hunt_python_api(self, fill_db):
        """Verify hunt command from python api parent"""
        version = fill_db
        if version < '0.1.8':
            pytest.skip("Python API not supported by {}".format(version))

        def function(x):
            """Evaluate partial information of a quadratic."""
            z = x - 34.56789
            return [dict(
                name='example_objective',
                type='objective',
                value=4 * z**2 + 23.4)]

        exp = create_experiment('hunt-python', branching={'branch-to': 'hunt-python-branch'})
        exp.workon(function, max_trials=10)

    # orion.core.cli.main('init-only') # TODO: deprecate init_only
