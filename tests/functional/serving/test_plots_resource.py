"""Perform tests for the REST endpoint `/plots`"""
import pytest

from orion.testing import falcon_client

config = dict(
    name="experiment-name",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "test-user",
        "orion_version": "XYZ",
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
    working_dir="",
    algorithm={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
)

trial_config = {
    "experiment": 0,
    "status": "completed",
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def test_root_not_available(client):
    """Tests that plots/regret is not available"""
    response = client.simulate_get("/plots/regret")

    assert response.status == "404 Not Found"


@pytest.mark.usefixtures("version_XYZ")
class TestRegretPlots:
    """Tests regret plots"""

    def test_unknown_experiment(self, client):
        """Tests that the API returns a 404 Not Found when an unknown experiment is queried."""
        response = client.simulate_get("/plots/regret/unknown-experiment")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_plot(self):
        """Tests that the API returns the plot in json format."""
        with falcon_client(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get("/plots/regret/experiment-name")

        assert response.status == "200 OK"
        assert response.json
        assert list(response.json.keys()) == ["data", "layout"]

    def test_no_trials(self):
        """Tests that the API returns an empty figure when no trials are found."""
        with falcon_client(config, trial_config, []) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get("/plots/regret/experiment-name")

        assert response.status == "200 OK"
        assert list(response.json.keys()) == ["data", "layout"]


@pytest.mark.usefixtures("version_XYZ")
class TestParallelCoordinatesPlots:
    """Tests parallel coordinates plots"""

    def test_unknown_experiment(self, client):
        """Tests that the API returns a 404 Not Found when an unknown experiment is queried."""
        response = client.simulate_get("/plots/parallel_coordinates/unknown-experiment")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_plot(self):
        """Tests that the API returns the plot in json format."""
        with falcon_client(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get(
                "/plots/parallel_coordinates/experiment-name"
            )

        assert response.status == "200 OK"
        assert response.json
        assert list(response.json.keys()) == ["data", "layout"]

    def test_no_trials(self):
        """Tests that the API returns an empty figure when no trials are found."""
        with falcon_client(config, trial_config, []) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get(
                "/plots/parallel_coordinates/experiment-name"
            )

        assert response.status == "200 OK"
        assert list(response.json.keys()) == ["data", "layout"]


@pytest.mark.usefixtures("version_XYZ")
class TestPartialDependenciesPlots:
    """Tests parallel coordinates plots"""

    def test_unknown_experiment(self, client):
        """Tests that the API returns a 404 Not Found when an unknown experiment is queried."""
        response = client.simulate_get("/plots/partial_dependencies/unknown-experiment")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_plot(self):
        """Tests that the API returns the plot in json format."""
        with falcon_client(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get(
                "/plots/partial_dependencies/experiment-name"
            )

        assert response.status == "200 OK"
        assert response.json
        assert list(response.json.keys()) == ["data", "layout"]

    def test_no_trials(self):
        """Tests that the API returns an empty figure when no trials are found."""
        with falcon_client(config, trial_config, []) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get(
                "/plots/partial_dependencies/experiment-name"
            )

        assert response.status == "200 OK"
        assert list(response.json.keys()) == ["data", "layout"]


@pytest.mark.usefixtures("version_XYZ")
class TestLPIPlots:
    """Tests lpi plots"""

    def test_unknown_experiment(self, client):
        """Tests that the API returns a 404 Not Found when an unknown experiment is queried."""
        response = client.simulate_get("/plots/lpi/unknown-experiment")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_plot(self):
        """Tests that the API returns the plot in json format."""
        with falcon_client(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get("/plots/lpi/experiment-name")

        assert response.status == "200 OK"
        assert response.json
        assert list(response.json.keys()) == ["data", "layout"]

    def test_no_trials(self):
        """Tests that the API returns an empty figure when no trials are found."""
        with falcon_client(config, trial_config, []) as (
            _,
            _,
            experiment,
            client,
        ):
            response = client.simulate_get("/plots/lpi/experiment-name")

        assert response.status == "200 OK"
        assert list(response.json.keys()) == ["data", "layout"]
