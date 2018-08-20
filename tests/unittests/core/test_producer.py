#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import copy

import pytest

from orion.core.worker.producer import Producer


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
    return Producer(hacked_exp)


def test_update(producer):
    """Test functionality of producer.update()."""
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


def test_update_and_produce(producer, database, random_dt):
    """Test functionality of producer.produce()."""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 1
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
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


def test_exceed_max_attempts(producer, database, random_dt):
    """Test that RuntimeError is raised when algo keep suggesting the same points"""
    producer.max_attempts = 10  # to limit run-time, default would work as well.
    producer.experiment.algorithms.algorithm.possible_values = [('rnn', 'rnn')]

    assert producer.experiment.pool_size == 1

    producer.update()

    with pytest.raises(RuntimeError) as exc_info:
        producer.produce()
    assert "Looks like the algorithm keeps suggesting" in str(exc_info.value)
