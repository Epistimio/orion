# -*- coding: utf-8 -*-
"""Perform functional tests for the REST endpoint `/experiments`"""
from orion.storage.base import get_storage

current_id = 0

config = dict(
    name='experiment-name',
    space={'x': 'uniform(0, 200)'},
    metadata={'user': 'test-user',
              'orion_version': 'XYZ',
              'VCS': {"type": "git",
                      "is_dirty": False,
                      "HEAD_sha": "test",
                      "active_branch": None,
                      "diff_sha": "diff"}},
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir='',
    algorithms={'random': {'seed': 1}},
    producer={'strategy': 'NoParallelStrategy'},
)


def _add_experiment(**kwargs):
    """Adds experiment to the dummy orion instance"""
    config.update(kwargs)
    get_storage().create_experiment(config)


def test_no_experiments(client):
    """Tests that the API returns a positive response when no experiments are present"""
    result = client.simulate_get('/experiments')

    assert result.json == []
    assert result.status == "200 OK"


def test_send_name_and_versions(client):
    """Tests that the API returns all the experiments with their name and version"""
    expected = [
        {'name': 'a', 'version': 1},
        {'name': 'b', 'version': 1}
    ]

    _add_experiment(name='a', version=1, _id=1)
    _add_experiment(name='b', version=1, _id=2)

    result = client.simulate_get('/experiments')

    assert result.json == expected
    assert result.status == "200 OK"


def test_latest_versions(client):
    """Tests that the API return the latest versions of each experiment"""
    expected = [
        {'name': 'a', 'version': 2},
        {'name': 'b', 'version': 1}
    ]

    _add_experiment(name='a', version=1, _id=1)
    _add_experiment(name='a', version=2, _id=2)
    _add_experiment(name='b', version=1, _id=3)

    result = client.simulate_get('/experiments')

    assert result.json == expected
    assert result.status == "200 OK"
