#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""


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
