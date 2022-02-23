#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures for functional serving tests"""
import pytest
from falcon import testing

from orion.serving.webapi import WebApi
from orion.testing import OrionState


@pytest.fixture()
def client():
    """Mock the falcon.API instance for testing with an in memory database"""
    storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
    with OrionState(storage=storage):
        yield testing.TestClient(WebApi({"storage": storage}))


@pytest.fixture()
def client_with_frontends_uri():
    """Mock the falcon.API instance for testing with custom frontend_uri"""
    storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
    with OrionState(storage=storage):
        yield testing.TestClient(
            WebApi(
                {
                    "storage": storage,
                    "frontends_uri": ["http://123.456", "http://example.com"],
                }
            )
        )
