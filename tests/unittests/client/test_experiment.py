#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.experiment`."""
from contextlib import contextmanager
import copy
import datetime
import logging

import pytest

import orion.core
from orion.client.experiment import ExperimentClient
import orion.core.io.experiment_builder as experiment_builder
from orion.core.io.database import DuplicateKeyError
from orion.core.worker.experiment import Experiment
from orion.core.worker.producer import Producer
from orion.core.worker.trial import Trial
from orion.core.utils.tests import OrionState


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
def new_config(script_path):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name='supernaekei',
        space={'x': 'uniform(0, 10)'},
        metadata={'user': 'tsirif',
                  'orion_version': 'XYZ',
                  'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "test",
                          "active_branch": None,
                          "diff_sha": "diff"}},
        version=1,
        pool_size=10,
        max_trials=1000,
        working_dir='',
        algorithms={
            'dumbalgo': {
                'done': False,
                'judgement': None,
                'scoring': 0,
                'seed': None,
                'suspend': False,
                'value': 5}},
        producer={'strategy': 'NoParallelStrategy'},
        refers=dict(
            root_id='supernaekei',
            parent_id=None,
            adapter=[])
        )

    return new_config


def _generate(obj, *args, value):
    if obj is None:
        return None

    obj = copy.deepcopy(obj)
    data = obj

    for arg in args[:-1]:
        data = data[arg]

    data[args[-1]] = value
    return obj


base_trial = {
    'experiment': 0,
    'status': 'new',  # new, reserved, suspended, completed, broken
    'worker': None,
    'start_time': None,
    'end_time': None,
    'heartbeat': None,
    'results': [],
    'params': []
}


def generate_trials(trial_config, status):
    """Generate Trials with different configurations"""
    new_trials = [_generate(trial_config, 'status', value=s) for s in status]

    for i, trial in enumerate(new_trials):
        trial['submit_time'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=i)
        if trial['status'] != 'new':
            trial['start_time'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=i)

    for i, trial in enumerate(new_trials):
        if trial['status'] == 'completed':
            trial['end_time'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=i)

    # make each trial unique
    for i, trial in enumerate(new_trials):
        if trial['status'] == 'completed':
            trial['results'].append({
                'name': 'loss',
                'type': 'objective',
                'value': i})

        trial['params'].append({
            'name': 'x',
            'type': 'real',
            'value': i
        })

    return new_trials


@contextmanager
def create_experiment(exp_config=None, trial_config=None, stati=None):
    if exp_config is None:
        exp_config = config
    if trial_config is None:
        trial_config = base_trial
    if stati is None:
        stati = ['new', 'interrupted', 'suspended', 'reserved', 'completed']

    with OrionState(experiments=[exp_config], trials=generate_trials(trial_config, stati)) as cfg:
        experiment = experiment_builder.build(name='supernaekei')
        if cfg.trials:
            experiment._id = cfg.trials[0]['experiment']
        client = ExperimentClient(experiment, Producer(experiment))
        yield cfg, experiment, client


def compare_trials(trials_a, trials_b):
    def to_dict(trial):
        return trial.to_dict()

    assert list(map(to_dict, trials_a)) == list(map(to_dict, trials_b))


def compare_without_heartbeat(trial_a, trial_b):

    trial_a_dict = trial_a.to_dict()
    trial_b_dict = trial_b.to_dict()
    trial_a_dict.pop('heartbeat')
    trial_b_dict.pop('heartbeat')
    assert trial_a_dict == trial_b_dict


def test_experiment_fetch_trials():
    with create_experiment() as (cfg, experiment, client):
        assert len(experiment.fetch_trials()) == 5
        compare_trials(experiment.fetch_trials(), client.fetch_trials())


def test_experiment_get_trial():
    with create_experiment() as (cfg, experiment, client):
        assert experiment.get_trial(uid=0) == client.get_trial(uid=0)


def test_experiment_fetch_trials_by_status():
    with create_experiment() as (cfg, experiment, client):
        compare_trials(experiment.fetch_trials_by_status('completed'),
                       client.fetch_trials_by_status('completed'))


def test_experiment_fetch_non_completed_trials():
    with create_experiment() as (cfg, experiment, client):
        compare_trials(experiment.fetch_noncompleted_trials(), client.fetch_noncompleted_trials())


class TestInsert:
    def test_insert_params_wo_results(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.insert(dict(x=100))
            assert trial.status == 'interrupted'
            assert trial.params[0].name == 'x'
            assert trial.params[0].value == 100
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_params_with_results(self):
        with create_experiment() as (cfg, experiment, client):
            timestamp = datetime.datetime.utcnow()
            trial = client.insert(dict(x=100), [dict(name='objective', type='objective', value=101)])
            assert trial.status == 'completed'
            assert trial.params[0].name == 'x'
            assert trial.params[0].value == 100
            assert trial.objective.value == 101
            assert trial.end_time >= timestamp
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))
            assert client.get_trial(uid=trial.id).objective.value == 101

            assert client._pacemakers == {}

    def test_insert_params_with_results_and_reserve(self):
        with create_experiment() as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                trial = client.insert(
                    dict(x=100),
                    [dict(name='objective', type='objective', value=101)],
                    reserve=True)

            assert 'Cannot observe a trial and reserve it' in str(exc.value)

    def test_insert_existing_params(self):
        with create_experiment() as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError) as exc:
                trial = client.insert(dict(x=1))

            assert ('A trial with params {\'x\': 1} already exist for experiment supernaekei-v1' ==
                    str(exc.value))

            assert client._pacemakers == {}

    def test_insert_partial_params(self):
        config_with_default = copy.deepcopy(config)
        config_with_default['space']['y'] = 'uniform(0, 10, default_value=5)'
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default['params'].append({'name': 'y', 'type': 'real', 'value': 1})
        with create_experiment(config_with_default, trial_with_default) as (cfg, experiment, client):
            trial = client.insert(dict(x=100))

            assert trial.status == 'interrupted'
            assert trial.params[0].name == 'x'
            assert trial.params[0].value == 100
            assert trial.params[1].name == 'y'
            assert trial.params[1].value == 5
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_partial_params_missing(self):
        config_with_default = copy.deepcopy(config)
        config_with_default['space']['y'] = 'uniform(0, 10)'
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default['params'].append({'name': 'y', 'type': 'real', 'value': 1})
        with create_experiment(config_with_default, trial_with_default) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x=1))

            assert ('Dimension y not specified and does not have a default value.' ==
                    str(exc.value))

    def test_insert_params_and_reserve(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.insert(dict(x=100), reserve=True)
            assert trial.status == 'reserved'
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_insert_params_fails_not_reserved(self):
        with create_experiment() as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError) as exc:
                client.insert(dict(x=1), reserve=True)

            assert client._pacemakers == {}

    def test_insert_bad_params(self):
        with create_experiment() as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x='bad bad bad'))

            assert 'Dimension x value bad bad bad is outside of prior uniform(0, 200)' == str(exc.value)
            assert client._pacemakers == {}

    def test_insert_params_bad_results(self):
        with create_experiment() as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x=100), [dict(name='objective', type='bad bad bad', value=0)])

            assert 'Given type, bad bad bad, not one of: ' in str(exc.value)
            assert client._pacemakers == {}


class TestReserve:
    def test_reserve(self):
        with create_experiment() as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]['_id'])
            assert trial.status != 'reserved'
            client.reserve(trial)
            assert trial.status == 'reserved'
            assert experiment.get_trial(trial).status == 'reserved'
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_reserve_dont_exist(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(experiment='idontexist', params=cfg.trials[0]['params'])
            with pytest.raises(ValueError) as exc:
                client.reserve(trial)

            assert 'Trial {} does not exist in database.'.format(trial.id) == str(exc.value)
            assert client._pacemakers == {}

    def test_reserve_reserved_locally(self, caplog):
        with create_experiment() as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]['_id'])
            assert trial.status != 'reserved'
            client.reserve(trial)
            with caplog.at_level(logging.WARNING):
                client.reserve(trial)

            assert 'Trial {} is already reserved.'.format(trial.id) == caplog.records[-1].message

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_reserve_reserved_remotely(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.status == 'interrupted'
            client.reserve(trial)
            remote_pacemaker = client._pacemakers.pop(trial.id)
            assert experiment.get_trial(trial).status == 'reserved'

            trial = Trial(**cfg.trials[1])
            assert trial.status == 'interrupted'
            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert 'Could not reserve trial {}.'.format(trial.id) == str(exc.value)

            assert trial.status == 'interrupted'
            assert experiment.get_trial(trial).status == 'reserved'
            assert client._pacemakers == {}
            remote_pacemaker.stop()

    def test_reserve_race_condition(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[0]['_id'])
            experiment.set_trial_status(trial, 'reserved')
            trial.status = 'new'  # Let's pretend it is still available
            
            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert 'Could not reserve trial {}.'.format(trial.id) == str(exc.value)
            assert client._pacemakers == {}


class TestRelease:
    def test_release(self):
        with create_experiment() as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]['_id'])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial)
            assert trial.status == 'interrupted'
            assert experiment.get_trial(trial).status == 'interrupted'
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_status(self):
        with create_experiment() as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]['_id'])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial, 'broken')
            assert trial.status == 'broken'
            assert experiment.get_trial(trial).status == 'broken'
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_dont_exist(self, monkeypatch):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(experiment='idontexist', params=cfg.trials[1]['params'])
            def do_nada(trial):
                return None
            monkeypatch.setattr(client, '_release_reservation', do_nada)

            with pytest.raises(ValueError) as exc:
                client.release(trial)

            assert 'Trial {} does not exist in database.'.format(trial.id) == str(exc.value)
            assert client._pacemakers == {}

    def test_release_race_condition(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]['_id'])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            # Woops! Trial got failed over from another process.
            experiment.set_trial_status(trial, 'interrupted')
            trial.status = 'reserved'  # Let's pretend we don't know.
            
            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert 'Reservation for trial {} has been lost before release.'.format(trial.id) in str(exc.value)
            assert client._pacemakers == {}
            assert not pacemaker.is_alive()

    def test_release_unreserved(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]['_id'])
            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert ('Trial {} had no pacemakers. Was is reserved properly?'.format(trial.id) ==
                    str(exc.value))

            assert client._pacemakers == {}


class TestSuggest:
    def test_suggest(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == 'reserved'
            assert trial.params[0].value == 1

            assert len(experiment.fetch_trials()) == 5
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_suggest_new(self):
        with create_experiment() as (cfg, experiment, client):
            for i in range(3):
                trial = client.suggest()
                assert trial.status == 'reserved'
                assert len(experiment.fetch_trials()) == 5
                assert client._pacemakers[trial.id].is_alive()
                client._pacemakers[trial.id].stop()

            trial = client.suggest()
            assert trial.status == 'reserved'
            assert trial.params[0].value == 57.567291697517554
            assert len(experiment.fetch_trials()) == 6

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_suggest_race_condition(self, monkeypatch):

        new_value = 50.

        # algo will suggest once an already existing trial
        def amnesia(num=1):
            if amnesia.count == 0:
                value = [0]
            else:
                value = [new_value]

            amnesia.count += 1

            return [value]

        amnesia.count = 0

        with create_experiment(stati=['completed']) as (cfg, experiment, client):

            monkeypatch.setattr(experiment.algorithms, 'suggest', amnesia)

            assert len(experiment.fetch_trials()) == 1

            trial = client.suggest()
            assert trial.status == 'reserved'
            assert trial.params[0].value == new_value
            assert amnesia.count == 2

            assert len(experiment.fetch_trials()) == 2
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_suggest_algo_opt_out(self, monkeypatch):
        def opt_out(num=1):
            return None

        orion.core.config.worker.max_idle_time = 0

        with create_experiment(stati=['completed']) as (cfg, experiment, client):

            monkeypatch.setattr(experiment.algorithms, 'suggest', opt_out)

            assert len(experiment.fetch_trials()) == 1

            assert client.suggest() is None

    def test_suggest_is_done(self):
        with create_experiment(stati=['completed'] * 10) as (cfg, experiment, client):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            assert client.suggest() is None

    def test_suggest_is_broken(self):
        with create_experiment(stati=['broken'] * 10) as (cfg, experiment, client):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_broken

            assert client.suggest() is None

    def test_suggest_is_done_race_condition(self, monkeypatch):
        """Verify that inability to suggest because is_done becomes True during produce() is
        handled."""
        with create_experiment(stati=['completed'] * 5) as (cfg, experiment, client):
            def is_done(self):
                return True

            def set_is_done():
                monkeypatch.setattr(experiment.__class__, 'is_done', property(is_done))

            monkeypatch.setattr(client._producer, 'produce', set_is_done)

            assert len(experiment.fetch_trials()) == 5
            assert not client.is_done

            assert client.suggest() is None

            assert len(experiment.fetch_trials()) == 5
            assert client.is_done

    def test_suggest_is_broken_race_condition(self, monkeypatch):
        with create_experiment(stati=['broken'] * 1) as (cfg, experiment, client):

            def is_broken(self):
                return True

            def set_is_broken():
                monkeypatch.setattr(experiment.__class__, 'is_broken', property(is_broken))

            monkeypatch.setattr(client._producer, 'produce', set_is_broken)

            assert len(experiment.fetch_trials()) == 1
            assert not client.is_broken

            assert client.suggest() is None

            assert len(experiment.fetch_trials()) == 1
            assert client.is_broken

class TestObserve:

    def test_observe(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.results == []
            client.reserve(trial)
            client.observe(trial, [dict(name='objective', type='objective', value=101)])

    def test_observe_unreserved(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            with pytest.raises(RuntimeError) as exc:
                client.observe(trial, [dict(name='objective', type='objective', value=101)])

            assert ('Trial {} had no pacemakers. Was is reserved properly?'.format(trial.id) ==
                    str(exc.value))

    def test_observe_dont_exist(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(experiment='idontexist', params=cfg.trials[0]['params'])
            with pytest.raises(ValueError) as exc:
                client.observe(trial, [dict(name='objective', type='objective', value=101)])

            assert 'Trial {} does not exist in database.'.format(trial.id) == str(exc.value)
            assert client._pacemakers == {}

    def test_observe_bad_results(self):
        with create_experiment() as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            client.reserve(trial)
            with pytest.raises(ValueError) as exc:
                client.observe(trial, [dict(name='objective', type='bad bad bad', value=101)])

            assert 'Given type, bad bad bad, not one of: ' in str(exc.value)
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers[trial.id].stop()

    def test_observe_race_condition(self):
        with create_experiment() as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]['_id'])
            client.reserve(trial)
            experiment.set_trial_status(trial, 'interrupted')
            trial.status = 'reserved'  # Let's pretend it is still reserved
            
            with pytest.raises(RuntimeError) as exc:
                client.observe(trial, [dict(name='objective', type='objective', value=101)])

            assert 'Reservation for trial {} has been lost.'.format(trial.id) == str(exc.value)
            assert client._pacemakers == {}


class TestWorkon:

    def test_workon(self):
        def foo(x):
            return x * 2

        with create_experiment(stati=[]) as (cfg, experiment, client):
            client.workon(foo, max_trials=10)
            assert len(experiment.fetch_trials()) == 10
            assert client._pacemakers == {}

    def test_workon_partial(self):
        assert False

    def test_workon_partial_with_override(self):
        # Params in kwargs are in trial.params
        assert False

    def test_workon_partial_with_override(self):
        # Params in kwargs are in trial.params
        assert False
