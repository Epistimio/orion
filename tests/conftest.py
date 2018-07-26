#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import os

from pymongo import MongoClient
import pytest
import yaml

from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.worker.trial import Trial
from orion.algo.dumbalgo import DumbAlgo


@pytest.fixture(scope='session')
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def categorical_values():
    """Return a list of all the categorical points possible for `supernaedo2` and `supernaedo3`"""
    return [('rnn', 'rnn'), ('rnn', 'lstm_with_attention'), ('rnn', 'gru'),
            ('gru', 'rnn'), ('gru', 'lstm_with_attention'), ('gru', 'gru'),
            ('lstm', 'rnn'), ('lstm', 'lstm_with_attention'), ('lstm', 'gru')]


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'unittests', 'core', 'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))

    for i, t_dict in enumerate(exp_config[1]):
        exp_config[1][i] = Trial(**t_dict).to_dict()

    return exp_config


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database, exp_config):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.experiments.insert_many(exp_config[0])
    database.trials.drop()
    database.trials.insert_many(exp_config[1])
    database.workers.drop()
    database.workers.insert_many(exp_config[2])
    database.resources.drop()
    database.resources.insert_many(exp_config[3])


@pytest.fixture()
def only_experiments_db(database, exp_config):
    """Clean the database and insert only experiments."""
    database.experiments.drop()
    database.experiments.insert_many(exp_config[0])
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    Database.instance = None
    MongoDB.instance = None
