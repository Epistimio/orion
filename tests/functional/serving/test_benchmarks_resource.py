"""Perform functional tests for the REST endpoint `/experiments`"""
import json
import os

import pytest
from falcon import testing

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import Branin, CarromTable, EggHolder
from orion.serving.webapi import WebApi
from orion.storage.base import setup_storage

current_id = 0


@pytest.fixture()
def client_with_benchmark():
    """Mock the falcon.API instance for testing with an in pickledDB database.

    If breaking changes in the benchmark causes issue with the saved database, simply
    delete the benchmark_db.pkl file and return the tests. The file will be recreated automatically.
    The new version of the file should be commit with git.
    """
    config_storage = {
        "type": "legacy",
        "database": {
            "type": "pickleddb",
            "host": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "benchmark_db.pkl"
            ),
        },
    }
    storage = setup_storage(config_storage)
    client = testing.TestClient(WebApi(storage, {"storage": config_storage}))

    _create_benchmark(storage)

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
                "assessments": {"AverageResult": {"repetitions": 2}},
                "tasks": {"Branin": {"max_trials": 10}},
            },
            {
                "name": "another_benchmark",
                "algorithms": ["gridsearch", {"random": {"seed": 1}}],
                "assessments": {
                    "AverageRank": {"repetitions": 2},
                    "AverageResult": {"repetitions": 2},
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

    def test_benchmark_specification(self, client_with_benchmark):
        """Tests that the benchmark returned is following the specification"""
        response = client_with_benchmark.simulate_get("/benchmarks/branin_baselines")

        assert response.status == "200 OK"

        assert response.json["name"] == "branin_baselines"
        assert response.json["algorithms"] == ["gridsearch", "random"]
        assert response.json["tasks"] == [{"Branin": {"max_trials": 10}}]
        assert response.json["assessments"] == [{"AverageResult": {"repetitions": 2}}]
        assert "AverageResult" in response.json["analysis"]
        assert "Branin" in response.json["analysis"]["AverageResult"]
        _assert_plot_contains(
            "gridsearch",
            response.json["analysis"]["AverageResult"]["Branin"]["regrets"],
        )
        _assert_plot_contains(
            "random", response.json["analysis"]["AverageResult"]["Branin"]["regrets"]
        )

    def test_benchmark_specific_assessment(self, client_with_benchmark):
        """Tests that the benchmark returned is following the specification"""
        response = client_with_benchmark.simulate_get("/benchmarks/another_benchmark")

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["algorithms"] == ["gridsearch", {"random": {"seed": 1}}]
        assert response.json["tasks"] == [
            {"Branin": {"max_trials": 10}},
            {"CarromTable": {"max_trials": 20}},
            {"EggHolder": {"dim": 4, "max_trials": 20}},
        ]
        assert response.json["assessments"] == [
            {"AverageResult": {"repetitions": 2}},
            {"AverageRank": {"repetitions": 2}},
        ]
        assert "AverageResult" in response.json["analysis"]
        assert "AverageRank" in response.json["analysis"]

        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?assessment=AverageResult"
        )

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["assessments"] == [
            {"AverageResult": {"repetitions": 2}},
            {"AverageRank": {"repetitions": 2}},
        ]
        assert "AverageResult" in response.json["analysis"]
        assert "AverageRank" not in response.json["analysis"]

    def test_benchmark_specific_task(self, client_with_benchmark):
        """Tests that the benchmark returned is following the specification"""
        response = client_with_benchmark.simulate_get("/benchmarks/another_benchmark")

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["algorithms"] == ["gridsearch", {"random": {"seed": 1}}]
        assert response.json["tasks"] == [
            {"Branin": {"max_trials": 10}},
            {"CarromTable": {"max_trials": 20}},
            {"EggHolder": {"dim": 4, "max_trials": 20}},
        ]
        assert response.json["assessments"] == [
            {"AverageResult": {"repetitions": 2}},
            {"AverageRank": {"repetitions": 2}},
        ]
        assert "Branin" in response.json["analysis"]["AverageResult"]
        assert "CarromTable" in response.json["analysis"]["AverageResult"]
        assert "EggHolder" in response.json["analysis"]["AverageResult"]

        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?task=CarromTable"
        )

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["tasks"] == [
            {"Branin": {"max_trials": 10}},
            {"CarromTable": {"max_trials": 20}},
            {"EggHolder": {"dim": 4, "max_trials": 20}},
        ]
        assert "Branin" not in response.json["analysis"]["AverageResult"]
        assert "CarromTable" in response.json["analysis"]["AverageResult"]
        assert "EggHolder" not in response.json["analysis"]["AverageResult"]

    def test_benchmark_specific_algorithms(self, client_with_benchmark):
        """Tests that the benchmark returned is following the specification"""
        response = client_with_benchmark.simulate_get("/benchmarks/another_benchmark")

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["algorithms"] == ["gridsearch", {"random": {"seed": 1}}]
        assert response.json["tasks"] == [
            {"Branin": {"max_trials": 10}},
            {"CarromTable": {"max_trials": 20}},
            {"EggHolder": {"dim": 4, "max_trials": 20}},
        ]
        assert response.json["assessments"] == [
            {"AverageResult": {"repetitions": 2}},
            {"AverageRank": {"repetitions": 2}},
        ]
        _assert_plot_contains(
            "gridsearch",
            response.json["analysis"]["AverageResult"]["Branin"]["regrets"],
        )
        _assert_plot_contains(
            "random", response.json["analysis"]["AverageResult"]["Branin"]["regrets"]
        )

        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?algorithms=random"
        )

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["algorithms"] == ["gridsearch", {"random": {"seed": 1}}]
        with pytest.raises(AssertionError):
            _assert_plot_contains(
                "gridsearch",
                response.json["analysis"]["AverageResult"]["Branin"]["regrets"],
            )
        _assert_plot_contains(
            "random", response.json["analysis"]["AverageResult"]["Branin"]["regrets"]
        )

        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?algorithms=random&algorithms=gridsearch"
        )

        assert response.status == "200 OK"
        assert response.json["name"] == "another_benchmark"
        assert response.json["algorithms"] == ["gridsearch", {"random": {"seed": 1}}]
        _assert_plot_contains(
            "gridsearch",
            response.json["analysis"]["AverageResult"]["Branin"]["regrets"],
        )
        _assert_plot_contains(
            "random", response.json["analysis"]["AverageResult"]["Branin"]["regrets"]
        )

    def test_benchmark_bad_assessment(self, client_with_benchmark):
        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?assessment=idontexist"
        )
        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Benchmark study not found",
            "description": (
                "Invalid assessment name: idontexist. "
                "It should be one of ['AverageRank', 'AverageResult']"
            ),
        }

    def test_benchmark_bad_task(self, client_with_benchmark):
        response = client_with_benchmark.simulate_get(
            "/benchmarks/another_benchmark?task=idontexist"
        )
        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Benchmark study not found",
            "description": (
                "Invalid task name: idontexist. "
                "It should be one of ['Branin', 'CarromTable', 'EggHolder']"
            ),
        }

    def test_benchmark_bad_algorithms(self, client_with_benchmark):
        response = client_with_benchmark.simulate_get(
            "/benchmarks/branin_baselines?algorithms=idontexist"
        )
        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Benchmark study not found",
            "description": (
                "Invalid algorithm: idontexist. "
                "It should be one of ['gridsearch', 'random']"
            ),
        }

    def test_benchmark_bad_algorithms_no_algorithm(self, client_with_benchmark):
        response = client_with_benchmark.simulate_get(
            "/benchmarks/branin_baselines?algorithms="
        )
        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Benchmark study not found",
            "description": (
                "Invalid algorithm: . It should be one of ['gridsearch', 'random']"
            ),
        }

    def test_unknown_parameter(self, client):
        """
        Tests that if an unknown parameter is specified in
        the query string, an error is returned even if the benchmark doesn't exist.
        """
        response = client.simulate_get("/benchmarks/a?unknown=true")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": (
                'Parameter "unknown" is not supported. '
                "Expected one of ['algorithms', 'assessment', 'task']."
            ),
        }


def _assert_plot_contains(legend_label, plotly_json):
    assert legend_label in [
        data_object.get("legendgroup", None)
        for data_object in json.loads(plotly_json)["data"]
    ]


def _create_benchmark(storage):
    benchmark = get_or_create_benchmark(
        storage=storage,
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
        storage=storage,
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
