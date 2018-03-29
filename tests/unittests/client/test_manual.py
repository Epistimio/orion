#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.manual`."""

import pytest

from orion.client import insert_trials
from orion.core.worker.experiment import Experiment


@pytest.mark.usefixtures("null_db_instances", "clean_db", "with_user_tsirif")
def test_good_insertion_exp_exists(monkeypatch, database, random_dt):
    """Check if a correct insertion to an existing experiment is done correctly."""
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    count_before = database.trials.count()
    insert_trials('supernaedo2', [('rnn', 'gru'), ('lstm', 'lstm_with_attention')])
    insert_trials('supernaedo3', [('rnn', 'rnn')])

    assert database.trials.count() == count_before + 3
    trials = list(database.trials.find())
    exp1 = Experiment('supernaedo2')
    exp2 = Experiment('supernaedo3')

    assert trials[-3]['experiment'] == exp1._id
    assert trials[-3]['status'] == 'new'
    assert trials[-3]['submit_time'] == random_dt
    assert len(trials[-3]['params']) == 2
    assert trials[-3]['params'][0] == {
        'name': '/encoding_layer',
        'type': 'categorical',
        'value': 'rnn'
        }
    assert trials[-3]['params'][1] == {
        'name': '/decoding_layer',
        'type': 'categorical',
        'value': 'gru'
        }

    assert trials[-2]['experiment'] == exp1._id
    assert trials[-2]['status'] == 'new'
    assert trials[-2]['submit_time'] == random_dt
    assert len(trials[-2]['params']) == 2
    assert trials[-2]['params'][0] == {
        'name': '/encoding_layer',
        'type': 'categorical',
        'value': 'lstm'
        }
    assert trials[-2]['params'][1] == {
        'name': '/decoding_layer',
        'type': 'categorical',
        'value': 'lstm_with_attention'
        }

    assert trials[-1]['experiment'] == exp2._id
    assert trials[-1]['status'] == 'new'
    assert trials[-1]['submit_time'] == random_dt
    assert len(trials[-1]['params']) == 2
    assert trials[-1]['params'][0] == {
        'name': '/encoding_layer',
        'type': 'categorical',
        'value': 'rnn'
        }
    assert trials[-1]['params'][1] == {
        'name': '/decoding_layer',
        'type': 'categorical',
        'value': 'rnn'
        }


@pytest.mark.usefixtures("null_db_instances", "clean_db", "with_user_tsirif")
def test_bad_insertion_exp_exists_ignores(database, random_dt):
    """Check if a bad insertion to an existing experiment will be ignored.

    The ones that are correct will be inserted.
    """
    config = dict(database={'name': 'orion_test',
                            'type': 'mongodb',
                            'host': 'mongodb://user:pass@localhost'})

    count_before = database.trials.count()
    insert_trials('supernaedo2', [('rnn', 5), ('lstm', 'lstm_with_attention')],
                  cmdconfig=config, raise_exc=False)
    insert_trials('supernaedo3', [(('lalal', 'ispii'), 'rnn')],
                  raise_exc=False)
    assert database.trials.count() == count_before + 1

    trials = list(database.trials.find())
    exp1 = Experiment('supernaedo2')

    assert trials[-1]['experiment'] == exp1._id
    assert trials[-1]['status'] == 'new'
    assert trials[-1]['submit_time'] == random_dt
    assert len(trials[-1]['params']) == 2
    assert trials[-1]['params'][0] == {
        'name': '/encoding_layer',
        'type': 'categorical',
        'value': 'lstm'
        }
    assert trials[-1]['params'][1] == {
        'name': '/decoding_layer',
        'type': 'categorical',
        'value': 'lstm_with_attention'
        }


@pytest.mark.usefixtures("null_db_instances", "clean_db", "with_user_tsirif")
def test_bad_insertion_exp_exists_raises(monkeypatch, database):
    """Check if a correct insertion to an existing experiment raises an exception.

    No set will be inserted
    """
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    count_before = database.trials.count()
    with pytest.raises(AssertionError):
        insert_trials('supernaedo2', [('rnn', 5), ('lstm', 'lstm_with_attention')],
                      raise_exc=True)
    with pytest.raises(AssertionError):
        insert_trials('supernaedo3', [(('lalal', 'ispii'), 'rnn')])
    assert database.trials.count() == count_before


@pytest.mark.usefixtures("null_db_instances", "clean_db", "with_user_tsirif")
def test_insertion_new_exp(monkeypatch, database):
    """Check that it raises if a new experiment name is given."""
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    count_before = database.trials.count()
    with pytest.raises(ValueError) as exc:
        insert_trials('supernaedo', [('rnn', 'rnn')])
    assert "No experiment named" in str(exc.value)
    assert database.trials.count() == count_before
