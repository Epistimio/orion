#!/usr/bin/env python
"""Common fixtures and utils for tests."""

import datetime
import getpass
import os

import pytest

from orion.algo.space import (Categorical, Integer, Real, Space)
from orion.core.io.convert import (JSONConverter, YAMLConverter)
from orion.core.io.database import Database
from orion.core.worker.experiment import Experiment

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_SAMPLE = os.path.join(TEST_DIR, 'sample_config.yml')
JSON_SAMPLE = os.path.join(TEST_DIR, 'sample_config.json')


@pytest.fixture()
def create_db_instance(null_db_instances, clean_db):
    """Create and save a singleton database instance."""
    database = Database(of_type='MongoDB', name='orion_test',
                        username='user', password='pass')
    return database


@pytest.fixture(scope='session')
def yaml_sample_path():
    """Return path with a yaml sample file."""
    return os.path.abspath(YAML_SAMPLE)


@pytest.fixture(scope='session')
def json_sample_path():
    """Return path with a json sample file."""
    return JSON_SAMPLE


@pytest.fixture(scope='session')
def yaml_converter():
    """Return a yaml converter."""
    return YAMLConverter()


@pytest.fixture(scope='session')
def json_converter():
    """Return a json converter."""
    return JSONConverter()


@pytest.fixture(scope='module')
def space():
    """Construct a simple space with every possible kind of Dimension."""
    space = Space()
    categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    dim = Categorical('yolo', categories, shape=2)
    space.register(dim)
    dim = Integer('yolo2', 'uniform', -3, 6)
    space.register(dim)
    dim = Real('yolo3', 'alpha', 0.9)
    space.register(dim)
    return space


@pytest.fixture(scope='module')
def fixed_suggestion():
    """Return the same tuple/sample from a possible space."""
    return (('asdfa', 2), 0, 3.5)


@pytest.fixture()
def with_user_tsirif(monkeypatch):
    """Make ``getpass.getuser()`` return ``'tsirif'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'tsirif')


@pytest.fixture()
def with_user_bouthilx(monkeypatch):
    """Make ``getpass.getuser()`` return ``'bouthilx'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'bouthilx')


@pytest.fixture()
def with_user_dendi(monkeypatch):
    """Make ``getpass.getuser()`` return ``'dendi'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'dendi')


@pytest.fixture()
def random_dt(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    random_dt = datetime.datetime(1903, 4, 25, 0, 0, 0)

    class MockDatetime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return random_dt

    monkeypatch.setattr(datetime, 'datetime', MockDatetime)
    return random_dt


@pytest.fixture()
def hacked_exp(with_user_dendi, random_dt, clean_db):
    """Return an `Experiment` instance with hacked _id to find trials in
    fake database.
    """
    try:
        Database(of_type='MongoDB', name='orion_test',
                 username='user', password='pass')
    except (TypeError, ValueError):
        pass
    exp = Experiment('supernaedo2')
    exp._id = 'supernaedo2'  # white box hack
    return exp


@pytest.fixture()
def trial_id_substitution(with_user_tsirif, random_dt, clean_db):
    """Replace trial ids by the actual ids of the experiments."""
    try:
        Database(of_type='MongoDB', name='orion_test',
                 username='user', password='pass')
    except (TypeError, ValueError):
        pass

    db = Database()
    experiments = db.read('experiments', {'metadata.user': 'tsirif'})
    experiment_dict = dict((experiment['name'], experiment) for experiment in experiments)
    trials = db.read('trials')

    for trial in trials:
        query = {'experiment': trial['experiment']}
        update = {'experiment': experiment_dict[trial['experiment']]['_id']}
        db.write('trials', update, query)
