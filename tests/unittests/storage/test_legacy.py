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

db_backends = [
    # {
    #     'type': 'PickledDB',
    #     'name': 'orion_test'
    # },
    # {
    #     'type': 'EphemeralDB',
    #     'name': 'orion_test'
    # },
    {
        'type': 'MongoDB',
        'name': 'orion_test',
        'username': 'user',
        'password': 'pass'
    }
]
