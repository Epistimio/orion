#!/usr/bin/env python
# -*- coding= utf-8 -*-
"""Example usage and tests for =mod:`orion.viz.analysers`."""
import copy

from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker.trial import Trial
from orion.viz.analysers.base import AnalyserWrapper

import pytest


@pytest.fixture
def exp(clean_db):
    config = dict(name="analysis_exp", user="tsirif", datetime="2017-11-22T20=00:00",
                  user_args=["", "-x~normal(0,10)", "-y~uniform(0,10)"], algorithms="random",
                  database=dict(type='mongodb', name='orion_test',
                                host='mongodb://user:pass@localhost'))

    return ExperimentBuilder().build_from(config)


@pytest.fixture
def base_trial(exp, random_dt):
    return dict(experiment=exp.id,
                results=[dict(name="example_objective", type="objective", value=0)],
                params=[dict(name="/x", type="real", value=0),
                        dict(name="/y", type="real", value=0)],
                status='completed', end_time=random_dt)


@pytest.fixture
def trials(base_trial, database):
    trials = []
    for i in range(3):
        trial = copy.deepcopy(base_trial)
        trial["results"][0]["value"] = i
        trial["params"][0]["value"] = i
        trial["params"][1]["value"] = i

        trials.append(trial)

    database.trials.drop()
    database.trials.insert_many(trials)

    trials = list(map(lambda x: Trial(**x), trials))
    return trials


@pytest.fixture
def lpi_config():
    return {'lpi': {'regressor_name': 'RandomForestRegressor',
                    'target_name': 'error_rate',
                    'target_args': ['example_objective'],
                    'n_trials_bootstraps': 'inf'}}


def test_commandline(trials, exp):
    analyser = AnalyserWrapper(trials, exp, {'commandline': {}})
    result = analyser.analyse()
    stats = exp.stats

    assert result.value == stats['best_trials_id']


def test_lpi(trials, exp, lpi_config):
    analyser = AnalyserWrapper(trials, exp, lpi_config)
    result = analyser.analyse().value

    assert len(result) == 2
    assert '/x' in result.keys()
    assert '/y' in result.keys()
