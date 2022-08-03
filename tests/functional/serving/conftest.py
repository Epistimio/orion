#!/usr/bin/env python
"""Common fixtures for functional serving tests"""
import pytest
from falcon import testing

from orion.serving.webapi import WebApi
from orion.testing import OrionState


@pytest.fixture()
def ephemeral_storage():
    storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
    with OrionState(storage=storage) as cfg:
        yield cfg


@pytest.fixture()
def client(ephemeral_storage):
    """Mock the falcon.API instance for testing with an in memory database"""
    yield testing.TestClient(WebApi(ephemeral_storage.storage, {}))


@pytest.fixture()
def client_with_frontends_uri(ephemeral_storage):
    """Mock the falcon.API instance for testing with custom frontend_uri"""
    yield testing.TestClient(
        WebApi(
            ephemeral_storage.storage,
            {
                "storage": ephemeral_storage.storage,
                "frontends_uri": ["http://123.456", "http://example.com"],
            },
        )
    )
