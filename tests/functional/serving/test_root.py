# -*- coding: utf-8 -*-
"""Perform functional tests for the REST endpoint `/`"""


def test_runtime_summary(client):
    """Tests if the instance's meta-summary is present"""
    response = client.simulate_get("/")

    assert response.json["orion"]
    assert response.json["server"] == "gunicorn"
    assert response.json["database"] == "EphemeralDB"
    assert response.status == "200 OK"
