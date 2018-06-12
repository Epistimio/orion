#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import pytest

from orion.core.worker.producer import Producer


@pytest.fixture()
def producer(hacked_exp, random_dt, exp_config):
    """Return a setup `Producer`."""
    # make init done
    hacked_exp.configure(exp_config[0][2])
    # insert fake point
    fake_point = ('gru', 'rnn')
    assert fake_point in hacked_exp.space
    hacked_exp.algorithms.algorithm.value = fake_point
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


@pytest.mark.skip(reason="DumbAlgo generates duplicate trials")
def test_update_and_produce(producer, database, random_dt):
    """Test functionality of producer.produce()."""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
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
         'value': 'lstm'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]
    assert new_trials[1]['experiment'] == producer.experiment.name
    assert new_trials[1]['start_time'] is None
    assert new_trials[1]['end_time'] is None
    assert new_trials[1]['results'] == []
    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'lstm'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]
