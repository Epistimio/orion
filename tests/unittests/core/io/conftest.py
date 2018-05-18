#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for io tests."""


import os

import pytest


@pytest.fixture()
def config_file():
    """Open config file with new config"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "orion_config.yaml")

    return open(file_path)


@pytest.fixture()
def old_config_file():
    """Open config file with original config from an experiment in db"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "orion_old_config.yaml")

    return open(file_path)


@pytest.fixture()
def incomplete_config_file():
    """Open config file with partial database configuration"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "orion_incomplete_config.yaml")

    return open(file_path)
