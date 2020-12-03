#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures for functional serving tests"""
from falcon import testing
import pytest

from orion.serving.webapi import WebApi
from orion.testing import OrionState


@pytest.fixture()
def client():
    """Mock the falcon.API instance for testing with an in memory database"""
    storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
    with OrionState(storage=storage):
        yield testing.TestClient(WebApi({"storage": storage}))
