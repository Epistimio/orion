# -*- coding: utf-8 -*-
"""
:mod:`orion.testing` -- Common testing support module
=====================================================
.. module:: testing
   :platform: Unix
   :synopsis: Common testing support module providing defaults, functions and mocks.
"""
# pylint: disable=protected-access

from contextlib import contextmanager
import copy
import datetime

import orion.core.io.experiment_builder as experiment_builder
from orion.core.worker.producer import Producer
from orion.testing.state import OrionState


def default_datetime():
    """Return default datetime"""
    return datetime.datetime(1903, 4, 25, 0, 0, 0)


def generate_trials(trial_config, statuses):
    """Generate Trials with different configurations"""

    def _generate(obj, *args, value):
        if obj is None:
            return None

        obj = copy.deepcopy(obj)
        data = obj

        data[args[-1]] = value
        return obj

    new_trials = [_generate(trial_config, 'status', value=s) for s in statuses]

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
def create_experiment(exp_config=None, trial_config=None, statuses=None):
    """Context manager for the creation of an ExperimentClient and storage init"""
    if exp_config is None:
        raise ValueError("Parameter 'exp_config' is missing")
    if trial_config is None:
        raise ValueError("Parameter 'trial_config' is missing")
    if statuses is None:
        statuses = ['new', 'interrupted', 'suspended', 'reserved', 'completed']

    from orion.client.experiment import ExperimentClient

    with OrionState(experiments=[exp_config],
                    trials=generate_trials(trial_config, statuses)) as cfg:
        experiment = experiment_builder.build(name=exp_config['name'])
        if cfg.trials:
            experiment._id = cfg.trials[0]['experiment']
        client = ExperimentClient(experiment, Producer(experiment))
        yield cfg, experiment, client

    client.close()


class MockDatetime(datetime.datetime):
    """Fake Datetime"""

    @classmethod
    def utcnow(cls):
        """Return our random/fixed datetime"""
        return default_datetime()
