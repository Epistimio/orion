#!/usr/bin/env python
"""Common fixtures and utils for tests."""

import os

from pymongo import MongoClient
import pytest
import yaml

from metaopt.io.database import Database
from metaopt.io.database.mongodb import MongoDB

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(TEST_DIR, 'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture(scope='module')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='metaopt_test')
    database = client.metaopt_test
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
def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    Database.instance = None
    MongoDB.instance = None


@pytest.fixture()
def create_db_instance(null_db_instances, clean_db):
    """Create and save a singleton database instance."""
    database = Database(of_type='MongoDB', name='metaopt_test',
                        username='user', password='pass')
    return database
