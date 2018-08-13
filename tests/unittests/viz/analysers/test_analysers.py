#!/usr/bin/env python
# -*- coding= utf-8 -*-
"""Example usage and tests for =mod:`orion.viz.analysers`."""
import copy

from orion.core.io.database import Database
from orion.core.utils import get_qualified_name
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.viz.analysers import AnalyserWrapper

import pytest


@pytest.fixture
def db():
    of_type = (get_qualified_name('orion.core.io.database', "mongodb"), "mongodb")
    db = Database(of_type=of_type, name="orion_test", host='mongodb://user:pass@localhost')
    return db


@pytest.fixture
def exp(db):
    config = dict(name="analysis_exp", user="tsirif", datetime="2017-11-22T20=00:00",
                  user_args=["-x~normal(0,1)", "-y~uniform(0,10)"], algorithms="random")

    return Experiment(config)


@pytest.fixture
def base_trial():
    return dict(experiment="analysis_exp", results=[dict(name="", type="objective", value=0)],
                params=[dict(name="/x", type="real", value=0),
                        dict(name="/y", type="real", value=0)])


@pytest.fixture
def trials(base_trial, db):
    trials = []
    for i in range(3):
        trial = copy.deepcopy(base_trial)
        trial["results"][0]["value"] = i
        trial["params"][0]["value"] = i
        trial["params"][1]["value"] = i

        trials.append(Trial(**trial))

    db.trials.drop()
    db.trials.insert_many(trials)
    return trials


def test_commandline(trials, exp):
    analyser = AnalyserWrapper(trials, exp, {'commandline': {}})
    result = analyser.analyse()

    print(result.value)
