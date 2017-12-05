#!/usr/bin/env python
"""Common fixtures and utils for tests."""

import os

import pytest
import yaml

from metaopt.io.database import Database
from metaopt.io.database.mongodb import MongoDB

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def exp_config():
    """Load an example database."""
    with open(os.path.join(TEST_DIR, 'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture()
def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    Database.instance = None
    MongoDB.instance = None
