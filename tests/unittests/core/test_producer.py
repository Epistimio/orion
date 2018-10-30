#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import copy

import pytest

from orion.core.worker.producer import Producer
from orion.core.worker.trial import Trial


class DumbParallelStrategy:
    """Mock object for parallel strategy"""

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        self._observed_points = points
        self._observed_results = results
        self._value = None

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if self._value:
            value = self._value
        else:
            value = len(self._observed_points)

        self._lie = lie = Trial.Result(name='lie', type='lie', value=value)
        return lie


@pytest.fixture()
def producer(hacked_exp, random_dt, exp_config, categorical_values):
    """Return a setup `Producer`."""
    # make init done

    # TODO: Remove this commented out if test pass
    # hacked_exp.configure(exp_config[0][3])
    # # insert fake point
    # fake_point = ('gru', 'rnn')
    # assert fake_point in hacked_exp.space
    # hacked_exp.algorithms.algorithm.value = fake_point

    hacked_exp.configure(exp_config[0][3])
    hacked_exp.pool_size = 1
    hacked_exp.algorithms.algorithm.possible_values = categorical_values

    hacked_exp.producer['strategy'] = DumbParallelStrategy()

    return Producer(hacked_exp)


def test_algo_observe_completed(producer):
    """Test that algo only observes completed trials"""
    assert len(producer.experiment.fetch_trials({})) > 3
    producer.update()
    # Algorithm must have received completed points and their results
    obs_points = producer.algorithm.algorithm._points
    obs_results = producer.algorithm.algorithm._results
    assert len(obs_points) == 3
    assert obs_points[0] == ('lstm', 'rnn')
    assert obs_points[1] == ('rnn', 'rnn')
    assert obs_points[2] == ('gru', 'lstm_with_attention')
    assert len(obs_results) == 3
    assert obs_results[0] == {
        'objective': 3,
        'gradient': None,
        'constraint': []
        }
    assert obs_results[1] == {
        'objective': 2,
        'gradient': (-0.1, 2),
        'constraint': []
        }
    assert obs_results[2] == {
        'objective': 10,
        'gradient': (5, 3),
        'constraint': [1.2]
        }


def test_strategist_observe_completed(producer):
    """Test that strategist only observes completed trials"""
    assert len(producer.experiment.fetch_trials({})) > 3
    producer.update()
    # Algorithm must have received completed points and their results
    obs_points = producer.strategy._observed_points
    obs_results = producer.strategy._observed_results
    assert len(obs_points) == 3
    assert obs_points[0] == ('lstm', 'rnn')
    assert obs_points[1] == ('rnn', 'rnn')
    assert obs_points[2] == ('gru', 'lstm_with_attention')
    assert len(obs_results) == 3
    assert obs_results[0] == {
        'objective': 3,
        'gradient': None,
        'constraint': []
        }
    assert obs_results[1] == {
        'objective': 2,
        'gradient': (-0.1, 2),
        'constraint': []
        }
    assert obs_results[2] == {
        'objective': 10,
        'gradient': (5, 3),
        'constraint': [1.2]
        }


def test_naive_algorithm_is_producing(producer, database, random_dt):
    """Verify naive algo is used to produce, not original algo"""
    producer.experiment.pool_size = 1
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'gru')]
    producer.update()
    producer.produce()

    assert producer.naive_algorithm.algorithm._num == 1  # pool size
    assert producer.algorithm.algorithm._num == 0


def test_update_and_produce(producer, database, random_dt):
    """Test new trials are properly produced"""
    possible_values = [('rnn', 'gru')]
    producer.experiment.pool_size = 1
    producer.experiment.algorithms.algorithm.possible_values = possible_values

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.naive_algorithm.algorithm._num
    assert num_new_points == 1  # pool size

    assert producer.naive_algorithm.algorithm._suggested == possible_values


def test_register_new_trials(producer, database, random_dt):
    """Test new trials are properly registered"""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 1
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.naive_algorithm.algorithm._num
    assert num_new_points == 1  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 1
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 1
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_no_lies_if_all_trials_completed(producer, database, random_dt):
    """Verify that no lies are created if all trials are completed"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    database.trials.remove(query)
    trials_in_db_before = database.trials.count({'experiment': producer.experiment.id})
    assert trials_in_db_before == 3

    producer.update()

    assert len(producer._produce_lies()) == 0


def test_lies_generation(producer, database, random_dt):
    """Verify that lies are created properly"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 4

    producer.update()

    lies = producer._produce_lies()
    assert len(lies) == 4

    for i in range(4):
        trials_non_completed[i]['_id'] = lies[i].id
        trials_non_completed[i]['status'] = 'completed'
        trials_non_completed[i]['end_time'] = random_dt
        trials_non_completed[i]['results'].append(producer.strategy._lie.to_dict())
        assert lies[i].to_dict() == trials_non_completed[i]


def test_naive_algo_not_trained_when_all_trials_completed(producer, database, random_dt):
    """Verify that naive algo is not trained on additional trials when all completed"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    database.trials.remove(query)
    trials_in_db_before = database.trials.count({'experiment': producer.experiment.id})
    assert trials_in_db_before == 3

    producer.update()

    assert len(producer.algorithm.algorithm._points) == 3
    assert len(producer.naive_algorithm.algorithm._points) == 3


def test_naive_algo_trained_on_all_non_completed_trials(producer, database, random_dt):
    """Verify that naive algo is trained on additional trials"""
    # Set two of completed trials to broken and reserved to have all possible status
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    completed_trials = database.trials.find(query)
    database.trials.update({'_id': completed_trials[0]['_id']}, {'$set': {'status': 'broken'}})
    database.trials.update({'_id': completed_trials[1]['_id']}, {'$set': {'status': 'reserved'}})

    # Make sure non completed trials and completed trials are set properly for the unit-test
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    non_completed_trials = list(database.trials.find(query))
    assert len(non_completed_trials) == 6
    # Make sure we have all type of status except completed
    assert (set(trial['status'] for trial in non_completed_trials) ==
            set(['new', 'reserved', 'suspended', 'interrupted', 'broken']))
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    assert database.trials.count(query) == 1

    # Executing the actual test
    producer.update()
    assert len(producer._produce_lies()) == 6

    assert len(producer.algorithm.algorithm._points) == 1
    assert len(producer.naive_algorithm.algorithm._points) == (1 + 6)


@pytest.mark.skip(reason="Waiting for rebase on non-blocking design PR...")
def test_concurent_producers(producer, database, random_dt):
    """Test concurrent production of new trials."""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    # Set so that first producer's algorithm generate valid point on first time
    # And second producer produce same point and thus must produce next one two.
    # Hence, we know that producer algo will have _num == 1 and
    # second producer algo will have _num == 2
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'gru'), ('gru', 'gru')]

    assert producer.experiment.pool_size == 1

    second_producer = Producer(producer.experiment)
    second_producer.algorithm = copy.deepcopy(producer.algorithm)

    producer.update()
    second_producer.update()

    producer.produce()
    second_producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
    assert num_new_points == 1  # pool size
    num_new_points = second_producer.algorithm.algorithm._num
    assert num_new_points == 2  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


@pytest.mark.skip(reason="Waiting for rebase on non-blocking design PR...")
def test_duplicate_within_pool(producer, database, random_dt):
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate.
    """
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 2

    producer.experiment.algorithms.algorithm.possible_values = [
        ('rnn', 'gru'), ('rnn', 'gru'), ('gru', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
    assert num_new_points == 4  # 2 * pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


@pytest.mark.skip(reason="Waiting for rebase on non-blocking design PR...")
def test_duplicate_within_pool_and_db(producer, database, random_dt):
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate with db.
    """
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 2

    producer.experiment.algorithms.algorithm.possible_values = [
        ('rnn', 'gru'), ('rnn', 'rnn'), ('gru', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
    assert num_new_points == 4  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


@pytest.mark.skip(reason="Waiting for rebase on non-blocking design PR...")
def test_exceed_max_attempts(producer, database, random_dt):
    """Test that RuntimeError is raised when algo keep suggesting the same points"""
    producer.max_attempts = 10  # to limit run-time, default would work as well.
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'rnn')]

    assert producer.experiment.pool_size == 1

    producer.update()

    with pytest.raises(RuntimeError) as exc_info:
        producer.produce()
    assert "Looks like the algorithm keeps suggesting" in str(exc_info.value)
