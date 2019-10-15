#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import copy
import datetime
import logging

import pytest

from orion.core.io.database import DuplicateKeyError
from orion.core.utils.tests import OrionState
from orion.core.worker.trial import Trial
from orion.storage.base import FailedUpdate, get_storage, MissingArguments
from orion.storage.track import HAS_TRACK, REASON

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

storage_backends = [
    None  # defaults to legacy with PickleDB
]

if not HAS_TRACK:
    log.warning('Track is not tested because: %s!', REASON)
else:
    storage_backends.append({
        'storage_type': 'track',
        'args': {
            'uri': 'file://${file}?objective=loss'
        }
    })

base_experiment = {
    'name': 'default_name',
    'version': 0,
    'metadata': {
        'user': 'default_user',
        'user_script': 'abc',
        'datetime': '2017-11-23T02:00:00'
    }
}


base_trial = {
    'experiment': 'default_name',
    'status': 'new',  # new, reserved, suspended, completed, broken
    'worker': None,
    'submit_time': '2017-11-23T02:00:00',
    'start_time': None,
    'end_time': None,
    'heartbeat': None,
    'results': [
        {'name': 'loss',
         'type': 'objective',  # objective, constraint
         'value': 2}
    ],
    'params': [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'lstm_with_attention'}
    ]
}


def _generate(obj, *args, value):
    if obj is None:
        return None

    obj = copy.deepcopy(obj)
    data = obj

    for arg in args[:-1]:
        data = data[arg]

    data[args[-1]] = value
    return obj


def make_lost_trial():
    """Make a lost trial"""
    obj = copy.deepcopy(base_trial)
    obj['status'] = 'reserved'
    obj['heartbeat'] = datetime.datetime.utcnow() - datetime.timedelta(seconds=61 * 2)
    obj['params'].append({
        'name': '/index',
        'type': 'categorical',
        'value': 'lost_trial'
    })
    return obj


all_status = ['completed', 'broken', 'reserved', 'interrupted', 'suspended', 'new']


def generate_trials(status=None):
    """Generate Trials with different configurations"""
    if status is None:
        status = all_status

    new_trials = [_generate(base_trial, 'status', value=s) for s in status]

    # make each trial unique
    for i, trial in enumerate(new_trials):
        trial['params'].append({
            'name': '/index',
            'type': 'categorical',
            'value': i
        })

    return new_trials


def generate_experiments():
    """Generate a set of experiments"""
    users = ['a', 'b', 'c']
    exps = [_generate(base_experiment, 'metadata', 'user', value=u) for u in users]
    return [_generate(exp, 'name', value=str(i)) for i, exp in enumerate(exps)]


@pytest.mark.parametrize('storage', storage_backends)
class TestStorage:
    """Test all storage backend"""

    def test_create_experiment(self, storage):
        """Test create experiment"""
        with OrionState(experiments=[], database=storage) as cfg:
            storage = cfg.storage()

            storage.create_experiment(base_experiment)

            experiments = storage.fetch_experiments({})
            assert len(experiments) == 1, 'Only one experiment in the database'

            experiment = experiments[0]
            assert base_experiment == experiment, 'Local experiment and DB should match'

            # Insert it again
            with pytest.raises(DuplicateKeyError):
                storage.create_experiment(base_experiment)

    def test_fetch_experiments(self, storage, name='0', user='a'):
        """Test fetch experiments"""
        with OrionState(experiments=generate_experiments(), database=storage) as cfg:
            storage = cfg.storage()

            experiments = storage.fetch_experiments({})
            assert len(experiments) == len(cfg.experiments)

            experiments = storage.fetch_experiments({'name': name, 'metadata.user': user})
            assert len(experiments) == 1

            experiment = experiments[0]
            assert experiment['name'] == name, 'name should match query'
            assert experiment['metadata']['user'] == user, 'user name should match query'

            experiments = storage.fetch_experiments({'name': '-1', 'metadata.user': user})
            assert len(experiments) == 0

    def test_update_experiment(self, monkeypatch, storage, name='0', user='a'):
        """Test fetch experiments"""
        with OrionState(experiments=generate_experiments(), database=storage) as cfg:
            storage = cfg.storage()

            class _Dummy():
                pass

            experiment = cfg.experiments[0]
            mocked_experiment = _Dummy()
            mocked_experiment._id = experiment['_id']

            print(experiment['_id'])
            storage.update_experiment(mocked_experiment, test=True)
            assert storage.fetch_experiments({'_id': experiment['_id']})[0]['test']
            assert 'test' not in storage.fetch_experiments({'_id': cfg.experiments[1]['_id']})[0]

            storage.update_experiment(uid=experiment['_id'], test2=True)
            assert storage.fetch_experiments({'_id': experiment['_id']})[0]['test2']
            assert 'test2' not in storage.fetch_experiments({'_id': cfg.experiments[1]['_id']})[0]

            with pytest.raises(MissingArguments):
                storage.update_experiment()

            with pytest.raises(AssertionError):
                storage.update_experiment(experiment=mocked_experiment, uid='123')

    def test_register_trial(self, storage):
        """Test register trial"""
        with OrionState(experiments=[base_experiment], database=storage) as cfg:
            storage = cfg.storage()
            trial1 = storage.register_trial(Trial(**base_trial))
            trial2 = storage.get_trial(trial1)

            assert trial1.to_dict() == trial2.to_dict(), 'Trials should match after insert'

    def test_register_duplicate_trial(self, storage):
        """Test register trial"""
        with OrionState(
                experiments=[base_experiment], trials=[base_trial], database=storage) as cfg:
            storage = cfg.storage()

            with pytest.raises(DuplicateKeyError):
                storage.register_trial(Trial(**base_trial))

    def test_register_lie(self, storage):
        """Test register lie"""
        with OrionState(experiments=[base_experiment], database=storage) as cfg:
            storage = cfg.storage()
            storage.register_lie(Trial(**base_trial))

    def test_register_lie_fail(self, storage):
        """Test register lie"""
        with OrionState(experiments=[base_experiment], lies=[base_trial], database=storage) as cfg:
            storage = cfg.storage()

            with pytest.raises(DuplicateKeyError):
                storage.register_lie(Trial(**cfg.lies[0]))

    def test_reserve_trial_success(self, storage):
        """Test reserve trial"""
        with OrionState(
                experiments=[base_experiment], trials=[base_trial], database=storage) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment('default_name', 'default_user', version=None)

            trial = storage.reserve_trial(experiment)
            assert trial is not None
            assert trial.status == 'reserved'

    def test_reserve_trial_fail(self, storage):
        """Test reserve trial"""
        with OrionState(
                experiments=[base_experiment],
                trials=generate_trials(status=['completed', 'reserved']),
                database=storage) as cfg:

            storage = cfg.storage()
            experiment = cfg.get_experiment('default_name', 'default_user', version=None)

            trial = storage.reserve_trial(experiment)
            assert trial is None

    def test_fetch_trials(self, storage):
        """Test fetch experiment trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment('default_name', 'default_user', version=None)

            trials1 = storage.fetch_trials(experiment=experiment)
            trials2 = storage.fetch_trials(uid=experiment._id)

            with pytest.raises(MissingArguments):
                storage.fetch_trials()

            with pytest.raises(AssertionError):
                storage.fetch_trials(experiment=experiment, uid='123')

            assert len(trials1) == len(cfg.trials), 'trial count should match'
            assert len(trials2) == len(cfg.trials), 'trial count should match'

    def test_get_trial(self, storage):
        """Test get trial"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            storage = cfg.storage()

            trial_dict = cfg.trials[0]

            trial1 = storage.get_trial(trial=Trial(**trial_dict))
            trial2 = storage.get_trial(uid=trial1.id)

            with pytest.raises(MissingArguments):
                storage.get_trial()

            with pytest.raises(AssertionError):
                storage.get_trial(trial=trial1, uid='123')

            assert trial1.to_dict() == trial_dict
            assert trial2.to_dict() == trial_dict

    def test_fetch_lost_trials(self, storage):
        """Test update heartbeat"""
        with OrionState(experiments=[base_experiment],
                        trials=generate_trials() + [make_lost_trial()], database=storage) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.fetch_lost_trials(experiment)

            count = 0
            now_datetime = datetime.datetime.utcnow()
            now_seconds = (now_datetime - datetime.datetime(1970, 1, 1)).total_seconds()
            for t in cfg.trials:
                status = t.get('status')
                if status == 'reserved':
                    heartbeat = t.get('heartbeat')
                    if heartbeat is None:
                        continue

                    diff = 0
                    if isinstance(heartbeat, datetime.datetime):
                        diff = (now_datetime - heartbeat).total_seconds()
                    else:
                        diff = now_seconds - heartbeat

                    if diff > 60 * 2:
                        count += 1

            assert len(trials) == count

    def test_change_status_success(self, storage):
        """Change the status of a Trial"""
        def check_status_change(new_status):
            with OrionState(
                    experiments=[base_experiment],
                    trials=generate_trials(), database=storage) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, 'was not able to retrieve trial for test'

                get_storage().set_trial_status(trial, status=new_status)
                assert trial.status == new_status, \
                    'Trial status should have been updated locally'

                trial = get_storage().get_trial(trial)
                assert trial.status == new_status, \
                    'Trial status should have been updated in the storage'

        check_status_change('completed')
        check_status_change('broken')
        check_status_change('reserved')
        check_status_change('interrupted')
        check_status_change('suspended')
        check_status_change('new')

    def test_change_status_failed_update(self, storage):
        """Change the status of a Trial"""
        def check_status_change(new_status):
            with OrionState(
                    experiments=[base_experiment],
                    trials=generate_trials(), database=storage) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, 'Was not able to retrieve trial for test'
                assert trial.status != new_status

                if trial.status == new_status:
                    return

                with pytest.raises(FailedUpdate):
                    trial.status = new_status
                    get_storage().set_trial_status(trial, status=new_status)

        check_status_change('completed')
        check_status_change('broken')
        check_status_change('reserved')
        check_status_change('interrupted')
        check_status_change('suspended')

    def test_fetch_pending_trials(self, storage):
        """Test fetch pending trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.fetch_pending_trials(experiment)

            count = 0
            for trial in cfg.trials:
                if trial['status'] in {'new', 'suspended', 'interrupted'}:
                    count += 1

            assert len(trials) == count
            for trial in trials:
                assert trial.status in {'new', 'suspended', 'interrupted'}

    def test_fetch_noncompleted_trials(self, storage):
        """Test fetch non completed trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.fetch_noncompleted_trials(experiment)

            count = 0
            for trial in cfg.trials:
                if trial['status'] != 'completed':
                    count += 1

            for trial in trials:
                assert trial.status != 'completed'

            assert len(trials) == count

    def test_fetch_trial_by_status(self, storage):
        """Test fetch completed trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial['status'] == 'completed':
                    count += 1

            storage = cfg.storage()
            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.fetch_trial_by_status(experiment, 'completed')

            assert len(trials) == count
            for trial in trials:
                assert trial.status == 'completed', trial

    def test_count_completed_trials(self, storage):
        """Test count completed trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial['status'] == 'completed':
                    count += 1

            storage = cfg.storage()

            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.count_completed_trials(experiment)
            assert trials == count

    def test_count_broken_trials(self, storage):
        """Test count broken trials"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial['status'] == 'broken':
                    count += 1

            storage = cfg.storage()

            experiment = cfg.get_experiment('default_name', 'default_user', version=None)
            trials = storage.count_broken_trials(experiment)

            assert trials == count

    def test_update_heartbeat(self, storage):
        """Test update heartbeat"""
        with OrionState(
                experiments=[base_experiment], trials=generate_trials(), database=storage) as cfg:
            storage_name = storage
            storage = cfg.storage()

            exp = cfg.get_experiment('default_name', 'default_user')
            trial1 = storage.fetch_trial_by_status(exp, status='reserved')[0]
            trial1b = copy.deepcopy(trial1)

            storage.update_heartbeat(trial1)

            trial2 = storage.get_trial(trial1)

            assert trial1b.heartbeat is None
            assert trial2.heartbeat is not None
            # this checks that heartbeat is the correct type and that it was updated prior to now
            assert trial2.heartbeat < datetime.datetime.utcnow()

            if storage_name is None:
                trial3 = storage.fetch_trial_by_status(exp, status='completed')[0]
                storage.update_heartbeat(trial3)

                assert trial3.heartbeat is None, \
                    'Legacy does not update trials with a status different from reserved'
