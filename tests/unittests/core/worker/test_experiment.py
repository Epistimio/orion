#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.experiment`."""

import copy
import datetime
import json
import tempfile

import pytest

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
import orion.core
from orion.core.evc.adapters import BaseAdapter
from orion.core.io.database import DuplicateKeyError
import orion.core.utils.backward as backward
from orion.core.utils.exceptions import RaceCondition
from orion.core.utils.tests import OrionState
import orion.core.worker.experiment
from orion.core.worker.experiment import Experiment, ExperimentView
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
        _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored='asdfa'
        )

    backward.populate_priors(new_config['metadata'])

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

    backward.populate_priors(config['metadata'])

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
    backward.populate_priors(config['metadata'])
    return config


def assert_protocol(exp, create_db_instance):
    """Transitional method to move away from mongodb"""
    assert exp._storage._db is create_db_instance


def count_experiment(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db.count("experiments")


def get_db_from_view(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db._db


@pytest.mark.usefixtures("create_db_instance", "with_user_tsirif")
class TestConfigProperty(object):
    """Get and set experiment's configuration, finilize initialization process."""

    @pytest.mark.usefixtures("trial_id_substitution")
    def test_status_is_pending_when_increase_max_trials(self, exp_config):
        """Attribute exp.algorithms become objects after init."""
        exp = Experiment('supernaedo4')

        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][3])

        assert exp.is_done

        exp = Experiment('supernaedo4')
        # Deliver an external configuration to finalize init
        exp_config[0][3]['max_trials'] = 1000
        exp.configure(exp_config[0][3])

        assert not exp.is_done


class TestReserveTrial(object):
    """Calls to interface `Experiment.reserve_trial`."""

    @pytest.mark.usefixtures("create_db_instance")
    def test_reserve_none(self):
        """Find nothing, return None."""
        exp = Experiment('supernaekei')
        trial = exp.reserve_trial()
        assert trial is None

    def test_reserve_success(self, exp_config_file, random_dt):
        """Successfully find new trials in db and reserve the first one"""
        with OrionState(from_yaml=exp_config_file) as cfg:
            exp = cfg.get_experiment('supernaedo2-dendi')
            trial = exp.reserve_trial()

            cfg.trials[1]['status'] = 'reserved'
            cfg.trials[1]['start_time'] = random_dt
            cfg.trials[1]['heartbeat'] = random_dt

            assert trial.to_dict() == cfg.trials[1]

    def test_reserve_when_exhausted(self, exp_config, hacked_exp):
        """Return None once all the trials have been allocated"""
        for _ in range(10):
            trial = hacked_exp.reserve_trial()

        assert trial is None

    def test_fix_lost_trials(self, hacked_exp, random_dt):
        """Test that a running trial with an old heartbeat is set to interrupted."""
        exp_query = {'experiment': hacked_exp.id}
        trial = hacked_exp.fetch_trials(exp_query)[0]
        heartbeat = random_dt - datetime.timedelta(seconds=180)

        get_storage().set_trial_status(trial, status='reserved', heartbeat=heartbeat)

        def fetch_trials(status='reserved'):
            trials = hacked_exp.fetch_trials_by_status(status)
            return list(filter(lambda new_trial: new_trial.id in [trial.id], trials))

        assert len(fetch_trials()) == 1

        hacked_exp.fix_lost_trials()

        assert len(fetch_trials()) == 0

        assert len(fetch_trials('interrupted')) == 1

    def test_fix_only_lost_trials(self, hacked_exp, random_dt):
        """Test that an old trial is set to interrupted but not a recent one."""
        exp_query = {'experiment': hacked_exp.id}
        trials = hacked_exp.fetch_trials(exp_query)
        lost = trials[0]
        not_lost = trials[1]

        heartbeat = random_dt - datetime.timedelta(seconds=180)

        get_storage().set_trial_status(lost, status='reserved', heartbeat=heartbeat)
        get_storage().set_trial_status(not_lost, status='reserved', heartbeat=random_dt)

        def fetch_trials():
            trials = hacked_exp.fetch_trials_by_status('reserved')
            return list(filter(lambda trial: trial.id in [lost.id, not_lost.id], trials))

        assert len(fetch_trials()) == 2

        hacked_exp.fix_lost_trials()

        assert len(fetch_trials()) == 1

        exp_query['status'] = 'interrupted'

        assert len(fetch_trials()) == 1

    def test_fix_lost_trials_race_condition(self, hacked_exp, random_dt, monkeypatch):
        """Test that a lost trial fixed by a concurrent process does not cause error."""
        exp_query = {'experiment': hacked_exp.id}
        trial = hacked_exp.fetch_trials(exp_query)[0]
        heartbeat = random_dt - datetime.timedelta(seconds=180)

        get_storage().set_trial_status(trial, status='interrupted', heartbeat=heartbeat)

        assert hacked_exp.fetch_trials(exp_query)[0].status == 'interrupted'

        def fetch_lost_trials(self, query):
            trial.status = 'reserved'
            return [trial]

        with monkeypatch.context() as m:
            m.setattr(hacked_exp.__class__, 'fetch_trials', fetch_lost_trials)
            hacked_exp.fix_lost_trials()

    def test_fix_lost_trials_configurable_hb(self, hacked_exp, random_dt):
        """Test that heartbeat is correctly being configured."""
        exp_query = {'experiment': hacked_exp.id}
        trial = hacked_exp.fetch_trials(exp_query)[0]
        old_heartbeat_value = orion.core.config.worker.heartbeat
        heartbeat = random_dt - datetime.timedelta(seconds=180)

        get_storage().set_trial_status(trial,
                                       status='reserved',
                                       heartbeat=heartbeat)

        trials = get_storage().fetch_trial_by_status(hacked_exp, 'reserved')

        assert trial.id in [t.id for t in trials]

        orion.core.config.worker.heartbeat = 210
        hacked_exp.fix_lost_trials()

        trials = get_storage().fetch_trial_by_status(hacked_exp, 'reserved')

        assert trial.id in [t.id for t in trials]

        orion.core.config.worker.heartbeat = old_heartbeat_value


def test_update_completed_trial(hacked_exp, database, random_dt):
    """Successfully push a completed trial into database."""
    trial = hacked_exp.reserve_trial()

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

    hacked_exp.update_completed_trial(trial, results_file=results_file)

    yo = database.trials.find_one({'_id': trial.id})

    assert len(yo['results']) == len(trial.results)
    assert yo['results'][0] == trial.results[0].to_dict()
    assert yo['status'] == 'completed'
    assert yo['end_time'] == random_dt

    results_file.close()


@pytest.mark.usefixtures("with_user_tsirif")
def test_register_trials(database, random_dt, hacked_exp):
    """Register a list of newly proposed trials/parameters."""
    hacked_exp._id = 'lalala'  # white box hack
    trials = [
        Trial(params=[{'name': 'a', 'type': 'integer', 'value': 5}]),
        Trial(params=[{'name': 'b', 'type': 'integer', 'value': 6}]),
        ]
    for trial in trials:
        hacked_exp.register_trial(trial)
    yo = list(database.trials.find({'experiment': hacked_exp._id}))
    assert len(yo) == len(trials)
    assert yo[0]['params'] == list(map(lambda x: x.to_dict(), trials[0].params))
    assert yo[1]['params'] == list(map(lambda x: x.to_dict(), trials[1].params))
    assert yo[0]['status'] == 'new'
    assert yo[1]['status'] == 'new'
    assert yo[0]['submit_time'] == random_dt
    assert yo[1]['submit_time'] == random_dt


def test_fetch_all_trials(hacked_exp, exp_config, random_dt):
    """Fetch a list of all trials"""
    query = dict()
    trials = hacked_exp.fetch_trials(query)
    sorted_exp_config = list(
        sorted(exp_config[1][0:7],
               key=lambda trial: trial.get('submit_time', datetime.datetime.utcnow())))

    assert len(trials) == 7
    for i in range(7):
        assert trials[i].to_dict() == sorted_exp_config[i]


def test_fetch_non_completed_trials(hacked_exp, exp_config):
    """Fetch a list of the trials that are not completed

    trials.status in ['new', 'interrupted', 'suspended', 'broken']
    """
    # Set two of completed trials to broken and reserved to have all possible status
    query = {'status': 'completed', 'experiment': hacked_exp.id}
    database = get_db_from_view(hacked_exp)
    completed_trials = database.trials.find(query)
    exp_config[1][0]['status'] = 'broken'
    database.trials.update({'_id': completed_trials[0]['_id']}, {'$set': {'status': 'broken'}})
    exp_config[1][2]['status'] = 'reserved'
    database.trials.update({'_id': completed_trials[1]['_id']}, {'$set': {'status': 'reserved'}})

    # Make sure non completed trials and completed trials are set properly for the unit-test
    query = {'status': {'$ne': 'completed'}, 'experiment': hacked_exp.id}
    non_completed_trials = list(database.trials.find(query))
    assert len(non_completed_trials) == 6
    # Make sure we have all type of status except completed
    assert (set(trial['status'] for trial in non_completed_trials) ==
            set(['new', 'reserved', 'suspended', 'interrupted', 'broken']))

    trials = hacked_exp.fetch_noncompleted_trials()
    assert len(trials) == 6

    def find_and_compare(trial_config):
        """Find the trial corresponding to given config and compare it"""
        trial = [trial for trial in trials if trial.id == trial_config['_id']]
        assert len(trial) == 1
        trial = trial[0]
        assert trial.to_dict() == trial_config

    find_and_compare(exp_config[1][0])
    find_and_compare(exp_config[1][2])
    find_and_compare(exp_config[1][3])
    find_and_compare(exp_config[1][4])
    find_and_compare(exp_config[1][5])
    find_and_compare(exp_config[1][6])


def test_is_done_property(hacked_exp):
    """Check experiment stopping conditions for maximum number of trials completed."""
    assert hacked_exp.is_done is False
    hacked_exp.max_trials = 2
    assert hacked_exp.is_done is True


def test_is_done_property_with_algo(hacked_exp):
    """Check experiment stopping conditions for algo which converged."""
    # Configure experiment to have instantiated algo
    hacked_exp.configure(hacked_exp.configuration)
    assert hacked_exp.is_done is False
    hacked_exp.algorithms.algorithm.done = True
    assert hacked_exp.is_done is True


def test_broken_property(hacked_exp):
    """Check experiment stopping conditions for maximum number of broken."""
    assert not hacked_exp.is_broken
    MAX_BROKEN = 3
    orion.core.config.worker.max_broken = MAX_BROKEN
    trials = hacked_exp.fetch_trials()[:MAX_BROKEN]

    for trial in trials:
        get_storage().set_trial_status(trial, status='broken')

    assert hacked_exp.is_broken


def test_configurable_broken_property(hacked_exp):
    """Check if max_broken changes after configuration."""
    assert not hacked_exp.is_broken
    MAX_BROKEN = 3
    orion.core.config.worker.max_broken = MAX_BROKEN
    trials = hacked_exp.fetch_trials()[:MAX_BROKEN]

    for trial in trials:
        get_storage().set_trial_status(trial, status='broken')

    assert hacked_exp.is_broken

    orion.core.config.worker.max_broken += 1

    assert not hacked_exp.is_broken


def test_experiment_stats(hacked_exp, exp_config, random_dt):
    """Check that property stats is returning a proper summary of experiment's results."""
    stats = hacked_exp.stats
    assert stats['trials_completed'] == 3
    assert stats['best_trials_id'] == exp_config[1][2]['_id']
    assert stats['best_evaluation'] == 2
    assert stats['start_time'] == exp_config[0][4]['metadata']['datetime']
    assert stats['finish_time'] == exp_config[1][1]['end_time']
    assert stats['duration'] == stats['finish_time'] - stats['start_time']
    assert len(stats) == 6


def test_fetch_completed_trials_from_view(hacked_exp, exp_config, random_dt):
    """Fetch a list of the unseen yet completed trials."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    trials = experiment_view.fetch_trials_by_status('completed')
    assert len(trials) == 3
    assert trials[0].to_dict() == exp_config[1][0]
    assert trials[1].to_dict() == exp_config[1][2]
    assert trials[2].to_dict() == exp_config[1][1]


def test_view_is_done_property(hacked_exp):
    """Check experiment stopping conditions accessed from view."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    # Fully configure wrapper experiment (should normally occur inside ExperimentView.__init__
    # but hacked_exp has been _hacked_ inside afterwards.
    hacked_exp.configure(hacked_exp.configuration)

    assert experiment_view.is_done is False

    with pytest.raises(AttributeError):
        experiment_view.max_trials = 2

    hacked_exp.max_trials = 2

    assert experiment_view.is_done is True


def test_view_algo_is_done_property(hacked_exp):
    """Check experiment's algo stopping conditions accessed from view."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    # Fully configure wrapper experiment (should normally occur inside ExperimentView.__init__
    # but hacked_exp has been _hacked_ inside afterwards.
    hacked_exp.configure(hacked_exp.configuration)

    assert experiment_view.is_done is False

    hacked_exp.algorithms.algorithm.done = True

    assert experiment_view.is_done is True


def test_experiment_view_stats(hacked_exp, exp_config, random_dt):
    """Check that property stats from view is consistent."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    stats = experiment_view.stats
    assert stats['trials_completed'] == 3
    assert stats['best_trials_id'] == exp_config[1][2]['_id']
    assert stats['best_evaluation'] == 2
    assert stats['start_time'] == exp_config[0][4]['metadata']['datetime']
    assert stats['finish_time'] == exp_config[1][1]['end_time']
    assert stats['duration'] == stats['finish_time'] - stats['start_time']
    assert len(stats) == 6


@pytest.mark.usefixtures("with_user_tsirif")
def test_experiment_view_protocol_read_only():
    """Verify that wrapper experiments' protocol is read-only"""
    exp = ExperimentView('supernaedo2')

    # Test that _protocol.set_trial_status indeed exists
    exp._experiment._storage._storage.set_trial_status

    with pytest.raises(AttributeError):
        exp._experiment._storage.set_trial_status
