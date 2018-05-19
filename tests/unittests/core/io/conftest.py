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
