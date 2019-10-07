#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import pytest

from orion.core.io.database import Database, DuplicateKeyError
from orion.core.utils.tests import OrionState


base_experiment = {
    'name': 'default_name',
    'version': 0,
    'metadata': {
        'user': 'default_user',
        'user_script': 'abc',
        'datetime': '2017-11-23T02:00:00'
    }
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


@pytest.mark.parametrize('db_backend', db_backends)
def test_backward_compatible_drop_user_index(db_backend):
    """Test that indexes from old versions are removed"""
    with OrionState(experiments=[], database=db_backend) as cfg:
        storage = cfg.storage()
        database = cfg.database

        database.ensure_index(
            'experiments',
            [('name', Database.ASCENDING), ('metadata.user', Database.ASCENDING)],
            unique=True)

        database.ensure_index(
            'experiments',
            [('name', Database.ASCENDING),
             ('metadata.user', Database.ASCENDING),
             ('version', Database.ASCENDING)],
            unique=True)

        storage.create_experiment(base_experiment)
        base_experiment.pop('_id')
        base_experiment['version'] = 1

        with pytest.raises(DuplicateKeyError):
            storage.create_experiment(base_experiment)

        experiments = storage.fetch_experiments({})
        assert len(experiments) == 1, 'Only first experiment in the database'

        # Remove old indexes for backward-compatibility
        storage._setup_db()  # pylint: disable=protected-access

        assert storage.create_experiment(base_experiment)

        experiments = storage.fetch_experiments({})
        assert len(experiments) == 2, 'Both experiments in the database'
