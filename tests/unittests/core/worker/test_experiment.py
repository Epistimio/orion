#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.experiment`."""

import copy
import datetime
import json
import logging
import tempfile

import pytest

import orion.core
from orion.core.io.space_builder import SpaceBuilder
import orion.core.utils.backward as backward
from orion.core.utils.tests import OrionState
import orion.core.worker.experiment
from orion.core.worker.experiment import Experiment, ExperimentView
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage


@pytest.fixture()
def new_config(random_dt):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name='supernaekei',
        # refers is missing on purpose
        metadata={'user': 'tsirif',
                  'orion_version': 0.1,
                  'user_script': 'abs_path/to_yoyoy.py',
                  'user_config': 'abs_path/hereitis.yaml',
                  'user_args': ['--mini-batch~uniform(32, 256, discrete=True)'],
                  'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "fsa7df7a8sdf7a8s7",
                          "active_branch": None,
                          "diff_sha": "diff"}},
        version=1,
        pool_size=10,
        max_trials=1000,
        working_dir=None,
        algorithms={'dumbalgo': {}},
        producer={'strategy': 'NoParallelStrategy'},
        # attrs starting with '_' also
        # _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored='asdfa'
        )

    backward.populate_space(new_config)

    return new_config


@pytest.fixture
def parent_version_config():
    """Return a configuration for an experiment."""
    config = dict(
        _id='parent_config',
        name="old_experiment",
        version=1,
        algorithms='random',
        metadata={'user': 'corneauf', 'datetime': datetime.datetime.utcnow(),
                  'user_args': ['--x~normal(0,1)']})

    backward.populate_space(config)

    return config


@pytest.fixture
def child_version_config(parent_version_config):
    """Return a configuration for an experiment."""
    config = copy.deepcopy(parent_version_config)
    config['_id'] = 'child_config'
    config['version'] = 2
    config['refers'] = {'parent_id': 'parent_config'}
    config['metadata']['datetime'] = datetime.datetime.utcnow()
    config['metadata']['user_args'].append('--y~+normal(0,1)')
    backward.populate_space(config)
    return config


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


def generate_trials(status):
    """Generate Trials with different configurations"""
    new_trials = [_generate(base_trial, 'status', value=s) for s in status]

    for i, trial in enumerate(new_trials):
        if trial['status'] != 'new':
            trial['start_time'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=i)

    for i, trial in enumerate(new_trials):
        if trial['status'] == 'completed':
            trial['end_time'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=i)

    # make each trial unique
    for i, trial in enumerate(new_trials):
        trial['results'][0]['value'] = i
        trial['params'].append({
            'name': '/index',
            'type': 'categorical',
            'value': i
        })

    return new_trials


def assert_protocol(exp, create_db_instance):
    """Transitional method to move away from mongodb"""
    assert exp._storage._db is create_db_instance


def count_experiment(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db.count("experiments")


def get_db_from_view(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db._db


@pytest.fixture()
def space():
    """Build a space object"""
    return SpaceBuilder().build({'x': 'uniform(0, 10)'})


@pytest.fixture()
def algorithm(space):
    """Build a dumb algo object"""
    return PrimaryAlgo(space, 'dumbalgo')


class TestReserveTrial(object):
    """Calls to interface `Experiment.reserve_trial`."""

    @pytest.mark.usefixtures("create_db_instance")
    def test_reserve_none(self):
        """Find nothing, return None."""
        with OrionState(experiments=[], trials=[]):
            exp = Experiment('supernaekei')
            trial = exp.reserve_trial()
            assert trial is None

    def test_reserve_success(self, random_dt):
        """Successfully find new trials in db and reserve the first one"""
        storage_config = {'type': 'legacy', 'database': {'type': 'EphemeralDB'}}
        with OrionState(trials=generate_trials(['new', 'reserved']),
                        storage=storage_config) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']

            trial = exp.reserve_trial()

            # Trials are sorted according to hash and 'new' gets position second
            cfg.trials[1]['status'] = 'reserved'
            cfg.trials[1]['start_time'] = random_dt
            cfg.trials[1]['heartbeat'] = random_dt

            assert trial.to_dict() == cfg.trials[1]

    def test_reserve_when_exhausted(self):
        """Return None once all the trials have been allocated"""
        stati = ['new', 'reserved', 'interrupted', 'completed', 'broken']
        with OrionState(trials=generate_trials(stati)) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']
            assert exp.reserve_trial() is not None
            assert exp.reserve_trial() is not None
            assert exp.reserve_trial() is None

    def test_fix_lost_trials(self):
        """Test that a running trial with an old heartbeat is set to interrupted."""
        trial = copy.deepcopy(base_trial)
        trial['status'] = 'reserved'
        trial['heartbeat'] = datetime.datetime.utcnow() - datetime.timedelta(seconds=360)
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']

            assert len(exp.fetch_trials_by_status('reserved')) == 1
            exp.fix_lost_trials()
            assert len(exp.fetch_trials_by_status('reserved')) == 0

    def test_fix_only_lost_trials(self):
        """Test that an old trial is set to interrupted but not a recent one."""
        lost_trial, running_trial = generate_trials(['reserved'] * 2)
        lost_trial['heartbeat'] = datetime.datetime.utcnow() - datetime.timedelta(seconds=360)
        running_trial['heartbeat'] = datetime.datetime.utcnow()

        with OrionState(trials=[lost_trial, running_trial]) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']

            assert len(exp.fetch_trials_by_status('reserved')) == 2

            exp.fix_lost_trials()

            reserved_trials = exp.fetch_trials_by_status('reserved')
            assert len(reserved_trials) == 1
            assert reserved_trials[0].to_dict()['params'] == running_trial['params']

            failedover_trials = exp.fetch_trials_by_status('interrupted')
            assert len(failedover_trials) == 1
            assert failedover_trials[0].to_dict()['params'] == lost_trial['params']

    def test_fix_lost_trials_race_condition(self, monkeypatch, caplog):
        """Test that a lost trial fixed by a concurrent process does not cause error."""
        trial = copy.deepcopy(base_trial)
        trial['status'] = 'interrupted'
        trial['heartbeat'] = datetime.datetime.utcnow() - datetime.timedelta(seconds=360)
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']

            assert len(exp.fetch_trials_by_status('interrupted')) == 1

            assert len(exp._storage.fetch_lost_trials(exp)) == 0

            def fetch_lost_trials(self, query):
                trial_object = Trial(**trial)
                trial_object.status = 'reserved'
                return [trial_object]

            # Force the fetch of a trial marked as reserved (and lost) while actually interrupted
            # (as if already failed-over by another process).
            with monkeypatch.context() as m:
                m.setattr(exp._storage.__class__, 'fetch_lost_trials', fetch_lost_trials)

                assert len(exp._storage.fetch_lost_trials(exp)) == 1

                with caplog.at_level(logging.DEBUG):
                    exp.fix_lost_trials()

            assert caplog.records[-1].levelname == 'DEBUG'
            assert caplog.records[-1].msg == 'failed'
            assert len(exp.fetch_trials_by_status('interrupted')) == 1
            assert len(exp.fetch_trials_by_status('reserved')) == 0

    def test_fix_lost_trials_configurable_hb(self):
        """Test that heartbeat is correctly being configured."""
        trial = copy.deepcopy(base_trial)
        trial['status'] = 'reserved'
        trial['heartbeat'] = datetime.datetime.utcnow() - datetime.timedelta(seconds=180)
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment('supernaekei')
            exp._id = cfg.trials[0]['experiment']

            assert len(exp.fetch_trials_by_status('reserved')) == 1

            orion.core.config.worker.heartbeat = 360

            exp.fix_lost_trials()

            assert len(exp.fetch_trials_by_status('reserved')) == 1

            orion.core.config.worker.heartbeat = 180

            exp.fix_lost_trials()

            assert len(exp.fetch_trials_by_status('reserved')) == 0


def test_update_completed_trial(random_dt):
    """Successfully push a completed trial into database."""
    with OrionState(trials=generate_trials(['new'])) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        trial = exp.reserve_trial()

        results_file = tempfile.NamedTemporaryFile(
            mode='w', prefix='results_', suffix='.log', dir='.', delete=True
        )

        # Generate fake result
        with open(results_file.name, 'w') as file:
            json.dump([{
                'name': 'loss',
                'type': 'objective',
                'value': 2}],
                file
            )
        # --

        exp.update_completed_trial(trial, results_file=results_file)

        yo = get_storage().fetch_trials(exp)[0].to_dict()

        assert len(yo['results']) == len(trial.results)
        assert yo['results'][0] == trial.results[0].to_dict()
        assert yo['status'] == 'completed'
        assert yo['end_time'] == random_dt

        results_file.close()


@pytest.mark.usefixtures("with_user_tsirif")
def test_register_trials(random_dt):
    """Register a list of newly proposed trials/parameters."""
    with OrionState():
        exp = Experiment('supernaekei')
        exp._id = 0

        trials = [
            Trial(params=[{'name': 'a', 'type': 'integer', 'value': 5}]),
            Trial(params=[{'name': 'b', 'type': 'integer', 'value': 6}]),
            ]
        for trial in trials:
            exp.register_trial(trial)

        yo = list(map(lambda trial: trial.to_dict(), get_storage().fetch_trials(exp)))
        assert len(yo) == len(trials)
        assert yo[0]['params'] == list(map(lambda x: x.to_dict(), trials[0]._params))
        assert yo[1]['params'] == list(map(lambda x: x.to_dict(), trials[1]._params))
        assert yo[0]['status'] == 'new'
        assert yo[1]['status'] == 'new'
        assert yo[0]['submit_time'] == random_dt
        assert yo[1]['submit_time'] == random_dt


def test_fetch_all_trials():
    """Fetch a list of all trials"""
    with OrionState(trials=generate_trials(['new', 'reserved', 'completed'])) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        trials = list(map(lambda trial: trial.to_dict(), exp.fetch_trials({})))
        assert trials == cfg.trials


def test_fetch_non_completed_trials():
    """Fetch a list of the trials that are not completed

    trials.status in ['new', 'interrupted', 'suspended', 'broken']
    """
    non_completed_stati = ['new', 'interrupted', 'suspended', 'reserved']
    stati = non_completed_stati + ['completed']
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        trials = exp.fetch_noncompleted_trials()
        assert len(trials) == 4
        assert set(trial.status for trial in trials) == set(non_completed_stati)


def test_is_done_property_with_pending(algorithm):
    """Check experiment stopping conditions when there is pending trials."""
    completed = ['completed'] * 10
    reserved = ['reserved'] * 5
    with OrionState(trials=generate_trials(completed + reserved)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        exp.algorithms = algorithm
        exp.max_trials = 10

        assert exp.is_done

        exp.max_trials = 15

        # There is only 10 completed trials
        assert not exp.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done but 5 trials are pending
        assert not exp.is_done


def test_is_done_property_no_pending(algorithm):
    """Check experiment stopping conditions when there is no pending trials."""
    completed = ['completed'] * 10
    broken = ['broken'] * 5
    with OrionState(trials=generate_trials(completed + broken)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        exp.algorithms = algorithm

        exp.max_trials = 15

        # There is only 10 completed trials and algo not done.
        assert not exp.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done and no pending trials
        assert exp.is_done


def test_broken_property():
    """Check experiment stopping conditions for maximum number of broken."""
    MAX_BROKEN = 3
    orion.core.config.worker.max_broken = MAX_BROKEN

    stati = (['reserved'] * 10) + (['broken'] * (MAX_BROKEN - 1))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        assert not exp.is_broken

    stati = (['reserved'] * 10) + (['broken'] * (MAX_BROKEN))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        assert exp.is_broken


def test_configurable_broken_property():
    """Check if max_broken changes after configuration."""
    MAX_BROKEN = 3
    orion.core.config.worker.max_broken = MAX_BROKEN

    stati = (['reserved'] * 10) + (['broken'] * (MAX_BROKEN))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']

        assert exp.is_broken

        orion.core.config.worker.max_broken += 1

        assert not exp.is_broken


def test_experiment_stats():
    """Check that property stats is returning a proper summary of experiment's results."""
    NUM_COMPLETED = 3
    stati = (['completed'] * NUM_COMPLETED) + (['reserved'] * 2)
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']
        exp.metadata = {'datetime': datetime.datetime.utcnow()}
        stats = exp.stats
        assert stats['trials_completed'] == NUM_COMPLETED
        assert stats['best_trials_id'] == cfg.trials[3]['_id']
        assert stats['best_evaluation'] == 0
        assert stats['start_time'] == exp.metadata['datetime']
        assert stats['finish_time'] == cfg.trials[0]['end_time']
        assert stats['duration'] == stats['finish_time'] - stats['start_time']
        assert len(stats) == 6


def test_fetch_completed_trials_from_view():
    """Fetch a list of the unseen yet completed trials."""
    non_completed_stati = ['new', 'interrupted', 'suspended', 'reserved']
    stati = non_completed_stati + ['completed']
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']
        exp_view = ExperimentView(exp)

        trials = exp_view.fetch_trials_by_status('completed')
        assert len(trials) == 1
        assert trials[0].status == 'completed'


def test_view_is_done_property_with_pending(algorithm):
    """Check experiment stopping conditions from view when there is pending trials."""
    completed = ['completed'] * 10
    reserved = ['reserved'] * 5
    with OrionState(trials=generate_trials(completed + reserved)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']
        exp.algorithms = algorithm
        exp.max_trials = 10

        exp_view = ExperimentView(exp)

        assert exp_view.is_done

        exp.max_trials = 15

        # There is only 10 completed trials
        assert not exp_view.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done but 5 trials are pending
        assert not exp_view.is_done


def test_view_is_done_property_no_pending(algorithm):
    """Check experiment stopping conditions from view when there is no pending trials."""
    completed = ['completed'] * 10
    broken = ['broken'] * 5
    with OrionState(trials=generate_trials(completed + broken)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']
        exp.algorithms = algorithm
        exp.max_trials = 100

        exp_view = ExperimentView(exp)

        exp.algorithms = algorithm

        exp.max_trials = 15

        # There is only 10 completed trials and algo not done.
        assert not exp_view.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done and no pending trials
        assert exp_view.is_done


def test_experiment_view_stats():
    """Check that property stats from view is consistent."""
    NUM_COMPLETED = 3
    stati = (['completed'] * NUM_COMPLETED) + (['reserved'] * 2)
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment('supernaekei')
        exp._id = cfg.trials[0]['experiment']
        exp.metadata = {'datetime': datetime.datetime.utcnow()}

        exp_view = ExperimentView(exp)

        stats = exp_view.stats
        assert stats['trials_completed'] == NUM_COMPLETED
        assert stats['best_trials_id'] == cfg.trials[3]['_id']
        assert stats['best_evaluation'] == 0
        assert stats['start_time'] == exp_view.metadata['datetime']
        assert stats['finish_time'] == cfg.trials[0]['end_time']
        assert stats['duration'] == stats['finish_time'] - stats['start_time']
        assert len(stats) == 6


def test_experiment_view_protocol_read_only():
    """Verify that wrapper experiments' protocol is read-only"""
    with OrionState():
        exp = Experiment('supernaekei')

        exp_view = ExperimentView(exp)

        # Test that _protocol.set_trial_status indeed exists
        exp_view._experiment._storage._storage.set_trial_status

        with pytest.raises(AttributeError):
            exp_view._experiment._storage.set_trial_status
