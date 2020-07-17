# -*- coding: utf-8 -*-
"""Perform functional tests for the REST endpoint `/`"""


def test_runtime_summary(client):
    """Tests if the instance's meta-summary is present"""
    result = client.simulate_get('/')

    assert "v" in result.json['orion']
    assert result.json['server'] == 'gunicorn'
    assert result.json['database'] == 'EphemeralDB'
    assert result.status == "200 OK"
