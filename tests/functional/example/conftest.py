#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for example tests."""

import os
import tempfile

import pytest


@pytest.fixture(scope="function")
def setup_database():
    """Configure the database"""
    temporary_file = tempfile.NamedTemporaryFile()

    os.environ['ORION_DB_TYPE'] = "pickleddb"
    os.environ['ORION_DB_ADDRESS'] = temporary_file.name
    yield
    temporary_file.close()
    os.environ.unsetenv("ORION_DB_TYPE")
    os.environ.unsetenv("ORION_DB_ADDRESS")
