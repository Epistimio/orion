#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for example tests."""

import os
import tempfile

import pytest


@pytest.fixture(scope="session")
def setup_database():
    """Configure the database"""
    os.environ['ORION_DB_TYPE'] = "pickleddb"
    os.environ['ORION_DB_ADDRESS'] = tempfile.NamedTemporaryFile().name
