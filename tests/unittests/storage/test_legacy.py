#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import json
import logging
import tempfile

import pytest

from orion.core.utils.tests import OrionState
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

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


mongodb_config = {
    'database': {
        'type': 'MongoDB',
        'name': 'orion_test',
        'username': 'user',
        'password': 'pass'
    }
}

db_backends = [
    {
        'storage_type': 'legacy',
        'args': {
            'config': mongodb_config
        }
    }
]


class TestLegacyStorage:
    """Test Legacy Storage retrieve result mechanic separately"""

    def test_push_trial_results(self, storage=None):
        """Successfully push a completed trial into database."""
        with OrionState(experiments=[], trials=[base_trial], database=storage) as cfg:
            storage = cfg.storage()
            trial = storage.get_trial(Trial(**base_trial))
            results = [
                Trial.Result(name='loss', type='objective', value=2)
            ]
            trial.results = results
            assert storage.push_trial_results(trial), 'should update successfully'

            trial2 = storage.get_trial(trial)
            assert trial2.results == results

    def retrieve_result(self, storage, generated_result):
        """Test retrieve result"""
        results_file = tempfile.NamedTemporaryFile(
            mode='w', prefix='results_', suffix='.log', dir='.', delete=True
        )

        # Generate fake result
        with open(results_file.name, 'w') as file:
            json.dump([generated_result], file)
        # --
        with OrionState(experiments=[], trials=[], database=storage) as cfg:
            storage = cfg.storage()

            trial = Trial(**base_trial)
            trial = storage.retrieve_result(trial, results_file)

            results = trial.results

            assert len(results) == 1
            assert results[0].to_dict() == generated_result

    def test_retrieve_result(self, storage=None):
        """Test retrieve result"""
        self.retrieve_result(storage, generated_result={
            'name': 'loss',
            'type': 'objective',
            'value': 2})

    def test_retrieve_result_incorrect_value(self, storage=None):
        """Test retrieve result"""
        with pytest.raises(ValueError) as exec:
            self.retrieve_result(storage, generated_result={
                'name': 'loss',
                'type': 'objective_unsupported_type',
                'value': 2})

        assert exec.match(r'Given type, objective_unsupported_type')

    def test_retrieve_result_nofile(self, storage=None):
        """Test retrieve result"""
        results_file = tempfile.NamedTemporaryFile(
            mode='w', prefix='results_', suffix='.log', dir='.', delete=True
        )

        with OrionState(experiments=[], trials=[], database=storage) as cfg:
            storage = cfg.storage()

            trial = Trial(**base_trial)

            with pytest.raises(json.decoder.JSONDecodeError) as exec:
                storage.retrieve_result(trial, results_file)

        assert exec.match(r'Expecting value: line 1 column 1 \(char 0\)')
