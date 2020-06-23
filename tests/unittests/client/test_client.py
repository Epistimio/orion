#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client`."""
import copy
from importlib import reload
import json

import pytest

from orion import client
import orion.core
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils import SingletonNotInstantiatedError
from orion.core.utils.exceptions import NoConfigurationError, RaceCondition
from orion.core.utils.tests import OrionState, update_singletons
from orion.storage.base import get_storage
from orion.storage.legacy import Legacy


create_experiment = client.create_experiment
workon = client.workon


config = dict(
    name='supernaekei',
    space={'x': 'uniform(0, 200)'},
    metadata={'user': 'tsirif',
              'orion_version': 'XYZ',
              'VCS': {"type": "git",
                      "is_dirty": False,
                      "HEAD_sha": "test",
                      "active_branch": None,
                      "diff_sha": "diff"}},
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir='',
    algorithms={'random': {'seed': 1}},
    producer={'strategy': 'NoParallelStrategy'},
    refers=dict(
        root_id='supernaekei',
        parent_id=None,
        adapter=[])
    )


@pytest.fixture()
def user_config():
    """Curate config as a user would provide it"""
    user_config = copy.deepcopy(config)
    user_config.pop('metadata')
    user_config.pop('version')
    user_config['strategy'] = user_config.pop('producer')['strategy']
    user_config.pop('refers')
    user_config.pop('pool_size')
    return user_config


@pytest.fixture()
def data():
    """Return serializable data."""
    return "this is datum"


class TestReportResults(object):
    """Check functionality and edge cases of `report_results` helper interface."""

    def test_with_no_env(self, monkeypatch, capsys, data):
        """Test without having set the appropriate environmental variable.

        Then: It should print `data` parameter instead to stdout.
        """
        monkeypatch.delenv('ORION_RESULTS_PATH', raising=False)
        reloaded_client = reload(client)

        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == data + '\n'
        assert err == ''

    def test_with_correct_env(self, monkeypatch, capsys, tmpdir, data):
        """Check that a file with correct data will be written to an existing
        file in a legit path.
        """
        path = str(tmpdir.join('naedw.txt'))
        with open(path, mode='w'):
            pass
        monkeypatch.setenv('ORION_RESULTS_PATH', path)
        reloaded_client = reload(client)

        assert reloaded_client.IS_ORION_ON is True
        assert reloaded_client.RESULTS_FILENAME == path
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == ''
        assert err == ''

        with open(path, mode='r') as results_file:
            res = json.load(results_file)
        assert res == data

    def test_with_env_set_but_no_file_exists(self, monkeypatch, tmpdir, data):
        """Check that a Warning will be raised at import time,
        if environmental is set but does not correspond to an existing file.
        """
        path = str(tmpdir.join('naedw.txt'))
        monkeypatch.setenv('ORION_RESULTS_PATH', path)

        with pytest.raises(RuntimeWarning) as exc:
            reload(client)

        assert "existing file" in str(exc.value)

    def test_call_interface_twice(self, monkeypatch, data):
        """Check that a Warning will be raised at call time,
        if function has already been called once.
        """
        monkeypatch.delenv('ORION_RESULTS_PATH', raising=False)
        reloaded_client = reload(client)

        reloaded_client.report_results(data)
        with pytest.raises(RuntimeWarning) as exc:
            reloaded_client.report_results(data)

        assert "already reported" in str(exc.value)
        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is True


class TestCreateExperiment:
    """Test creation of experiment with `client.create_experiment()`"""

    def test_create_experiment_no_storage(self):
        """Test creation if storage is not configured"""
        name = 'oopsie_forgot_a_storage'
        storage_type = 'legacy'
        db_type = 'pickleddb'
        host = 'test.pkl'
        orion.core.config.storage.type = storage_type
        orion.core.config.storage.database.type = db_type
        orion.core.config.storage.database.host = host

        with OrionState(storage=orion.core.config.storage.to_dict()) as cfg:
            # Reset the Storage and drop instances so that get_storage() would fail.
            cfg.cleanup()
            cfg.singletons = update_singletons()

            # Make sure storage must be instantiated during `create_experiment()`
            with pytest.raises(SingletonNotInstantiatedError):
                get_storage()

            experiment = create_experiment(name=name, space={'x': 'uniform(0, 10)'})

            assert isinstance(experiment._experiment._storage, Legacy)
            assert isinstance(experiment._experiment._storage._db, PickledDB)
            assert experiment._experiment._storage._db.host == host

    def test_create_experiment_new_no_space(self):
        """Test that new experiment needs space"""
        with OrionState():
            name = 'oopsie_forgot_a_space'
            with pytest.raises(NoConfigurationError) as exc:
                create_experiment(name=name)

            assert 'Experiment {} does not exist in DB'.format(name) in str(exc.value)

    def test_create_experiment_bad_storage(self):
        """Test error message if storage is not configured properly"""
        name = 'oopsie_bad_storage'
        # Make sure there is no existing storage singleton
        update_singletons()

        with pytest.raises(NotImplementedError) as exc:
            create_experiment(name=name, storage={'type': 'legacy',
                                                  'database': {'type': 'idontexist'}})

        assert "Could not find implementation of AbstractDB, type = 'idontexist'" in str(exc.value)

    def test_create_experiment_new_default(self):
        """Test creating a new experiment with all defaults"""
        name = 'all_default'
        space = {'x': 'uniform(0, 10)'}
        with OrionState():
            experiment = create_experiment(name='all_default', space=space)

            assert experiment.name == name
            assert experiment.space.configuration == space

            assert experiment.max_trials == orion.core.config.experiment.max_trials
            assert experiment.working_dir == orion.core.config.experiment.working_dir
            assert experiment.algorithms.configuration == {'random': {'seed': None}}
            assert experiment.configuration['producer'] == {'strategy': 'MaxParallelStrategy'}

    def test_create_experiment_new_full_config(self, user_config):
        """Test creating a new experiment by specifying all attributes."""
        with OrionState():
            experiment = create_experiment(**user_config)

            exp_config = experiment.configuration

            assert exp_config['space'] == config['space']
            assert exp_config['max_trials'] == config['max_trials']
            assert exp_config['working_dir'] == config['working_dir']
            assert exp_config['algorithms'] == config['algorithms']
            assert exp_config['producer'] == config['producer']

    def test_create_experiment_hit_no_branch(self, user_config):
        """Test creating an existing experiment by specifying all identical attributes."""
        with OrionState(experiments=[config]):
            experiment = create_experiment(**user_config)

            exp_config = experiment.configuration

            assert experiment.name == config['name']
            assert experiment.version == 1
            assert exp_config['space'] == config['space']
            assert exp_config['max_trials'] == config['max_trials']
            assert exp_config['working_dir'] == config['working_dir']
            assert exp_config['algorithms'] == config['algorithms']
            assert exp_config['producer'] == config['producer']

    def test_create_experiment_hit_no_config(self):
        """Test creating an existing experiment by specifying the name only."""
        with OrionState(experiments=[config]):
            experiment = create_experiment(config['name'])

            assert experiment.name == config['name']
            assert experiment.version == 1
            assert experiment.space.configuration == config['space']
            assert experiment.algorithms.configuration == config['algorithms']
            assert experiment.max_trials == config['max_trials']
            assert experiment.working_dir == config['working_dir']
            assert experiment.producer['strategy'].configuration == config['producer']['strategy']

    def test_create_experiment_hit_branch(self):
        """Test creating a differing experiment that cause branching."""
        with OrionState(experiments=[config]):
            experiment = create_experiment(config['name'], space={'y': 'uniform(0, 10)'})

            assert experiment.name == config['name']
            assert experiment.version == 2

            assert experiment.algorithms.configuration == config['algorithms']
            assert experiment.max_trials == config['max_trials']
            assert experiment.working_dir == config['working_dir']
            assert experiment.producer['strategy'].configuration == config['producer']['strategy']

    def test_create_experiment_race_condition(self, monkeypatch):
        """Test that a single race condition is handled seemlessly

        RaceCondition during registration is already handled by `build()`, therefore we will only
        test for race conditions during version update.
        """
        with OrionState(experiments=[config]):
            parent = create_experiment(config['name'])
            child = create_experiment(config['name'], space={'y': 'uniform(0, 10)'})

            def insert_race_condition(self, query):
                is_auto_version_query = (
                    query == {'name': config['name'], 'refers.parent_id': parent.id})
                if is_auto_version_query:
                    data = [child.configuration]
                # First time the query returns no other child
                elif insert_race_condition.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition.count += int(is_auto_version_query)

                return data

            insert_race_condition.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition)

            experiment = create_experiment(config['name'], space={'y': 'uniform(0, 10)'})

            assert insert_race_condition.count == 1
            assert experiment.version == 2
            assert experiment.configuration == child.configuration

    def test_create_experiment_race_condition_broken(self, monkeypatch):
        """Test that two or more race condition leads to raise"""
        with OrionState(experiments=[config]):
            parent = create_experiment(config['name'])
            child = create_experiment(config['name'], space={'y': 'uniform(0, 10)'})

            def insert_race_condition(self, query):
                is_auto_version_query = (
                    query == {'name': config['name'], 'refers.parent_id': parent.id})
                if is_auto_version_query:
                    data = [child.configuration]
                # The query returns no other child, never!
                else:
                    data = [parent.configuration]

                insert_race_condition.count += int(is_auto_version_query)

                return data

            insert_race_condition.count = 0

            monkeypatch.setattr(get_storage().__class__, 'fetch_experiments',
                                insert_race_condition)

            with pytest.raises(RaceCondition) as exc:
                create_experiment(config['name'], space={'y': 'uniform(0, 10)'})

            assert insert_race_condition.count == 2
            assert 'There was a race condition during branching and new version' in str(exc.value)

    def test_create_experiment_hit_manual_branch(self):
        """Test creating a differing experiment that cause branching."""
        new_space = {'y': 'uniform(0, 10)'}
        with OrionState(experiments=[config]):
            create_experiment(config['name'], space=new_space)

            with pytest.raises(ValueError) as exc:
                create_experiment(config['name'], version=1, space=new_space)

            assert "Configuration is different and generates" in str(exc.value)


class TestWorkon:
    """Test the helper function for sequential API"""

    def test_workon(self):
        """Verify that workon processes properly"""
        def foo(x):
            return [dict(name='result', type='objective', value=x * 2)]

        experiment = workon(foo, space={'x': 'uniform(0, 10)'}, max_trials=5)
        assert len(experiment.fetch_trials()) == 5
        assert experiment.name == 'loop'
        assert isinstance(experiment._experiment._storage, Legacy)
        assert isinstance(experiment._experiment._storage._db, EphemeralDB)

    def test_workon_algo(self):
        """Verify that algo config is processed properly"""
        def foo(x):
            return [dict(name='result', type='objective', value=x * 2)]

        experiment = workon(
            foo, space={'x': 'uniform(0, 10)'}, max_trials=5,
            algorithms={'random': {'seed': 5}})

        assert experiment.algorithms.algorithm.seed == 5

    def test_workon_name(self):
        """Verify setting the name with workon"""
        def foo(x):
            return [dict(name='result', type='objective', value=x * 2)]

        experiment = workon(foo, space={'x': 'uniform(0, 10)'}, max_trials=5, name='voici')

        assert experiment.name == 'voici'

    def test_workon_twice(self):
        """Verify setting the each experiment has its own storage"""
        def foo(x):
            return [dict(name='result', type='objective', value=x * 2)]

        experiment = workon(foo, space={'x': 'uniform(0, 10)'}, max_trials=5, name='voici')

        assert experiment.name == 'voici'
        assert len(experiment.fetch_trials()) == 5

        experiment2 = workon(foo, space={'x': 'uniform(0, 10)'}, max_trials=1, name='voici')

        assert experiment2.name == 'voici'
        assert len(experiment2.fetch_trials()) == 1
