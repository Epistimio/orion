#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import os

from pymongo import MongoClient
import pytest
import yaml

from orion.algo.base import (BaseAlgorithm, OptimizationAlgorithm)
import orion.core.cli
from orion.core.io.database import Database


class DumbAlgo(BaseAlgorithm):
    """Stab class for `BaseAlgorithm`."""

    def __init__(self, space, value=5,
                 scoring=0, judgement=None,
                 suspend=False, done=False, **nested_algo):
        """Configure returns, allow for variable variables."""
        self._times_called_suspend = 0
        self._times_called_is_done = 0
        self._num = None
        self._points = None
        self._results = None
        self._score_point = None
        self._judge_point = None
        self._measurements = None
        super(DumbAlgo, self).__init__(space, value=value,
                                       scoring=scoring, judgement=judgement,
                                       suspend=suspend,
                                       done=done,
                                       **nested_algo)

    def suggest(self, num=1):
        """Suggest based on `value`."""
        self._num = num
        return [self.value] * num

    def observe(self, points, results):
        """Log inputs."""
        self._points = points
        self._results = results

    def score(self, point):
        """Log and return stab."""
        self._score_point = point
        return self.scoring

    def judge(self, point, measurements):
        """Log and return stab."""
        self._judge_point = point
        self._measurements = measurements
        return self.judgement

    @property
    def should_suspend(self):
        """Cound how many times it has been called and return `suspend`."""
        self._times_called_suspend += 1
        return self.suspend

    @property
    def is_done(self):
        """Cound how many times it has been called and return `done`."""
        self._times_called_is_done += 1
        return self.done


# Hack it into being discoverable
OptimizationAlgorithm.types.append(DumbAlgo)
OptimizationAlgorithm.typenames.append(DumbAlgo.__name__.lower())


@pytest.fixture(scope='session')
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database, db_instance):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.lying_trials.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def db_instance(null_db_instances):
    """Create and save a singleton database instance."""
    try:
        db = Database(of_type='MongoDB', name='orion_test',
                      username='user', password='pass')
    except ValueError:
        db = Database()

    return db


@pytest.fixture
@pytest.mark.usefixtures('clean_db')
def only_experiments_db(exp_config):
    """Clean the database and insert only experiments."""
    database.experiments.insert_many(exp_config[0])


# Experiments combinations fixtures
@pytest.fixture
def one_experiment(monkeypatch, db_instance):
    """Create a single experiment."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_single_exp',
                         './black_box.py', '--x~uniform(0,1)'])


@pytest.fixture
def two_experiments(monkeypatch, db_instance):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         './black_box.py', '--x~uniform(0,1)'])
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         '--branch', 'test_double_exp_child', './black_box.py',
                         '--x~uniform(0,1)', '--y~+uniform(0,1)'])


@pytest.fixture
def three_experiments(monkeypatch, two_experiments, one_experiment):
    """Create a single experiment and an experiment and its child."""
    pass
