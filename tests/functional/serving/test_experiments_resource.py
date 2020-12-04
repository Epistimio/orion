# -*- coding: utf-8 -*-
"""Perform functional tests for the REST endpoint `/experiments`"""
import copy
import datetime

from orion.core.worker.trial import Trial
from orion.storage.base import get_storage

current_id = 0

base_experiment = dict(
    name="experiment-name",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "test-user",
        "orion_version": "x.y.z",
        "datetime": datetime.datetime(1, 1, 1),
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    pool_size=1,
    max_trials=10,
    max_broken=7,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
)

base_trial = {
    "experiment": None,
    "status": "new",
    "worker": None,
    "submit_time": datetime.datetime(1, 1, 1),
    "start_time": datetime.datetime(1, 1, 1, second=10),
    "end_time": datetime.datetime(1, 1, 2),
    "heartbeat": None,
    "results": [
        {"name": "loss", "type": "objective", "value": 0.05},
        {"name": "a", "type": "statistic", "value": 10},
        {"name": "b", "type": "statistic", "value": 5},
    ],
    "params": [
        {"name": "x", "type": "real", "value": 10.0},
    ],
}


class TestCollection:
    """Tests the server's response on experiments/"""

    def test_no_experiments(self, client):
        """Tests that the API returns a positive response when no experiments are present"""
        response = client.simulate_get("/experiments")

        assert response.json == []
        assert response.status == "200 OK"

    def test_send_name_and_versions(self, client):
        """Tests that the API returns all the experiments with their name and version"""
        expected = [{"name": "a", "version": 1}, {"name": "b", "version": 1}]

        _add_experiment(name="a", version=1, _id=1)
        _add_experiment(name="b", version=1, _id=2)

        response = client.simulate_get("/experiments")

        assert response.json == expected
        assert response.status == "200 OK"

    def test_latest_versions(self, client):
        """Tests that the API return the latest versions of each experiment"""
        expected = [{"name": "a", "version": 3}, {"name": "b", "version": 1}]

        _add_experiment(name="a", version=1, _id=1)
        _add_experiment(name="a", version=3, _id=2)
        _add_experiment(name="a", version=2, _id=3)
        _add_experiment(name="b", version=1, _id=4)

        response = client.simulate_get("/experiments")

        assert response.json == expected
        assert response.status == "200 OK"


class TestItem:
    """Tests the server's response to experiments/:name"""

    def test_non_existent_experiment(self, client):
        """
        Tests that a 404 response is returned when the experiment
        doesn't exist in the database
        """
        response = client.simulate_get("/experiments/a")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "a" does not exist',
        }

        _add_experiment(name="a", version=1, _id=1)
        response = client.simulate_get("/experiments/b")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "b" does not exist',
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
    assert config["poolSize"] == 1
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
