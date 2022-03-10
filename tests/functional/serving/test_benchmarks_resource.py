# -*- coding: utf-8 -*-
"""Perform functional tests for the REST endpoint `/experiments`"""
import copy
import datetime
import os

import pytest
from falcon import testing

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import Branin, CarromTable, EggHolder, RosenBrock, profet
from orion.serving.webapi import WebApi
from orion.storage.base import get_storage


current_id = 0


@pytest.fixture()
def client_with_benchmark():
    """Mock the falcon.API instance for testing with an in pickledDB database.

    If breaking changes in the benchmark causes issue with the saved database, simply
    delete the benchmark_db.pkl file and return the tests. The file will be recreated automatically.
    The new version of the file should be commit with git.
    """
    storage = {
        "type": "legacy",
        "database": {
            "type": "pickleddb",
            "host": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "benchmark_db.pkl"
            ),
        },
    }
    client = testing.TestClient(WebApi({"storage": storage}))

    if not get_storage().fetch_benchmark({}):
        _create_benchmark()

    return client


class TestCollection:
    """Tests the server's response on benchmarks/"""

    def test_no_benchmarks(self, client):
        """Tests that the API returns a positive response when no benchmarks are present"""
        response = client.simulate_get("/benchmarks")

        assert response.json == []
        assert response.status == "200 OK"

    def test_send_configuration(self, client_with_benchmark):
        """Tests that the API returns all the benchmarks with their configuration"""
        response = client_with_benchmark.simulate_get("/benchmarks")

        assert response.json == [
            {
                "name": "branin_baselines",
                "algorithms": ["gridsearch", "random"],
                "assessments": {"AverageResult": {"task_num": 2}},
                "tasks": {"Branin": {"max_trials": 10}},
            },
            {
                "name": "another_benchmark",
                "algorithms": ["gridsearch", {"random": {"seed": 1}}],
                "assessments": {
                    "AverageRank": {"task_num": 2},
                    "AverageResult": {"task_num": 2},
                },
                "tasks": {
                    "Branin": {"max_trials": 10},
                    "CarromTable": {"max_trials": 20},
                    "EggHolder": {"dim": 4, "max_trials": 20},
                },
            },
        ]
        assert response.status == "200 OK"


class TestItem:
    """Tests the server's response to benchmarks/:name"""

    def test_non_existent_benchmark(self, client):
        """
        Tests that a 404 response is returned when the benchmark
        doesn't exist in the database
        """
        response = client.simulate_get("/benchmarks/a")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Benchmark not found",
            "description": 'Benchmark "a" does not exist',
        }

    def test_experiment_specification(self, client):
        """Tests that the experiment returned is following the specification"""
        _add_experiment(name="a", version=1, _id=1)
        _add_trial(experiment=1, id_override="ae8", status="completed")

        response = client.simulate_get("/experiments/a")

        assert response.status == "200 OK"

        assert response.json["name"] == "a"
        assert response.json["version"] == 1
        assert response.json["status"] == "not done"
        assert response.json["trialsCompleted"] == 1
        assert response.json["startTime"] == "0001-01-01 00:00:00"  # TODO
        assert response.json["endTime"] == "0001-01-02 00:00:00"  # TODO
        assert len(response.json["user"])
        assert response.json["orionVersion"] == "x.y.z"

        _assert_config(response.json["config"])
        _assert_best_trial(response.json["bestTrial"])

    def test_default_is_latest_version(self, client):
        """Tests that the latest experiment is returned when no version parameter exists"""
        _add_experiment(name="a", version=1, _id=1)
        _add_experiment(name="a", version=3, _id=2)
        _add_experiment(name="a", version=2, _id=3)

        response = client.simulate_get("/experiments/a")

        assert response.status == "200 OK"
        assert response.json["version"] == 3

    def test_specific_version(self, client):
        """Tests that the specified version of an experiment is returned"""
        _add_experiment(name="a", version=1, _id=1)
        _add_experiment(name="a", version=2, _id=2)
        _add_experiment(name="a", version=3, _id=3)

        response = client.simulate_get("/experiments/a?version=2")

        assert response.status == "200 OK"
        assert response.json["version"] == 2

    def test_unknown_parameter(self, client):
        """
        Tests that if an unknown parameter is specified in
        the query string, an error is returned even if the experiment doesn't exist.
        """
        response = client.simulate_get("/experiments/a?unknown=true")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": 'Parameter "unknown" is not supported. Expected parameter "version".',
        }

        _add_experiment(name="a", version=1, _id=1)

        response = client.simulate_get("/experiments/a?unknown=true")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": 'Parameter "unknown" is not supported. Expected parameter "version".',
        }


def _add_experiment(**kwargs):
    """Adds experiment to the dummy orion instance"""
    base_experiment.update(copy.deepcopy(kwargs))
    get_storage().create_experiment(base_experiment)


def _add_trial(**kwargs):
    """Add trials to the dummy orion instance"""
    base_trial.update(copy.deepcopy(kwargs))
    get_storage().register_trial(Trial(**base_trial))


def _assert_config(config):
    """Asserts properties of the ``config`` dictionary"""
    assert config["maxTrials"] == 10
    assert config["maxBroken"] == 7

    algorithm = config["algorithm"]
    assert algorithm["name"] == "random"
    assert algorithm["seed"] == 1

    space = config["space"]
    assert len(space) == 1
    assert space["x"] == "uniform(0, 200)"


def _assert_best_trial(best_trial):
    """Verifies properties of the best trial"""
    assert best_trial["id"] == "ae8"
    assert best_trial["submitTime"] == "0001-01-01 00:00:00"
    assert best_trial["startTime"] == "0001-01-01 00:00:10"
    assert best_trial["endTime"] == "0001-01-02 00:00:00"

    parameters = best_trial["parameters"]
    assert len(parameters) == 1
    assert parameters["x"] == 10.0

    assert best_trial["objective"] == 0.05

    statistics = best_trial["statistics"]
    assert statistics["a"] == 10
    assert statistics["b"] == 5


def _create_benchmark():
    benchmark = get_or_create_benchmark(
        name="branin_baselines",
        algorithms=["gridsearch", "random"],
        targets=[
            {
                "assess": [AverageResult(2)],
                "task": [Branin(max_trials=10)],
            }
        ],
    )

    # Execute the benchmark
    benchmark.process()
    import pprint

    pprint.pprint(benchmark.configuration)

    benchmark = get_or_create_benchmark(
        name="another_benchmark",
        algorithms=["gridsearch", {"random": {"seed": 1}}],
        targets=[
            {
                "assess": [AverageResult(2), AverageRank(2)],
                "task": [Branin(max_trials=10), CarromTable(20), EggHolder(20, dim=4)],
            }
        ],
    )

    # Execute the benchmark
    benchmark.process()

    pprint.pprint(benchmark.configuration)
