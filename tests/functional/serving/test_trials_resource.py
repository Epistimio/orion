"""Perform tests for the REST endpoint `/trials`"""
import copy
import datetime
import random

import falcon

from orion.core.io import experiment_builder
from orion.core.worker.trial import Trial

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
    refers={},
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir="",
    algorithms={"random": {"seed": 1}},
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


def add_experiment(storage, **kwargs):
    """Adds experiment to the dummy orion instance"""
    base_experiment.update(copy.deepcopy(kwargs))
    experiment_builder.build(
        branching=dict(branch_from=base_experiment["name"], enable=True),
        storage=storage,
        **base_experiment
    )


def add_trial(storage, experiment: int, status: str = None, value=10, **kwargs):
    """
    Add trials to the dummy orion instance

    Parameters
    ----------
    experiment
        The ID of the experiment (stored in Experiment._id)
    status
        The status of the trial to add.
    **kwargs
        Any other attribute to modify from the base trial config.
    """
    if not status:
        status = random.choice(Trial.allowed_stati)

    kwargs["experiment"] = experiment
    kwargs["status"] = status

    base_trial.update(copy.deepcopy(kwargs))
    base_trial["params"][0]["value"] = value
    storage.register_trial(Trial(**base_trial))


def test_root_endpoint_not_supported(client):
    """Tests that the server return a 404 when accessing trials/ with no experiment"""
    response = client.simulate_get("/trials")

    assert response.status == "404 Not Found"
    if falcon.__version__ < "3.0.0":
        assert not response.json
    else:
        assert response.json == {"title": "404 Not Found"}


class TestTrialCollection:
    """Tests trials/:experiment_name"""

    def test_trials_for_unknown_experiment(self, client):
        """Tests that an unknown experiment returns a not found error"""
        response = client.simulate_get("/trials/unknown-experiment")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_unknown_parameter(self, client, ephemeral_storage):
        """
        Tests that if an unknown parameter is specified in
        the query string, an error is returned even if the experiment doesn't exist.
        """
        storage = ephemeral_storage.storage

        expected_error_message = (
            'Parameter "unknown" is not supported. '
            "Expected one of ['ancestors', 'status', 'version']."
        )

        response = client.simulate_get("/trials/a?unknown=true")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": expected_error_message,
        }

        add_experiment(storage, name="a", version=1, _id=1)

        response = client.simulate_get("/trials/a?unknown=true")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": expected_error_message,
        }

    def test_trials_for_latest_version(self, client, ephemeral_storage):
        """Tests that it returns the trials of the latest version of the experiment"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)
        add_experiment(storage, name="a", version=2, _id=2)

        add_trial(storage, experiment=1, id_override="00", value=10)
        add_trial(storage, experiment=2, id_override="01", value=10)
        add_trial(storage, experiment=1, id_override="02", value=11)
        add_trial(storage, experiment=2, id_override="03", value=12)

        response = client.simulate_get("/trials/a")

        assert response.status == "200 OK"
        assert response.json == [
            {"id": "49aeacec1eeaf290fd5fa84bf8f3874f"},
            {"id": "36b6bb34f0a01764e1793fe2d4de9078"},
        ]

    def test_trials_for_specific_version(self, client, ephemeral_storage):
        """Tests specific version of experiment"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)
        add_experiment(storage, name="a", version=2, _id=2)
        add_experiment(storage, name="a", version=3, _id=3)

        add_trial(storage, experiment=1, id_override="00")
        add_trial(storage, experiment=2, id_override="01")
        add_trial(storage, experiment=3, id_override="02")

        # Happy case
        response = client.simulate_get("/trials/a?version=2")

        assert response.status == "200 OK"
        assert response.json == [{"id": "49aeacec1eeaf290fd5fa84bf8f3874f"}]

        # Version doesn't exist
        response = client.simulate_get("/trials/a?version=4")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "a" has no version "4"',
        }

    def test_trials_for_all_versions(self, client, ephemeral_storage):
        """Tests that trials from all ancestors are shown"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)
        add_experiment(storage, name="a", version=2, _id=2)
        add_experiment(storage, name="a", version=3, _id=3)

        # Specify values to avoid duplicates
        add_trial(storage, experiment=1, id_override="00", value=1)
        add_trial(storage, experiment=2, id_override="01", value=2)
        add_trial(storage, experiment=3, id_override="02", value=3)

        # Happy case default
        response = client.simulate_get("/trials/a?ancestors=true")

        assert response.status == "200 OK"
        assert response.json == [
            {"id": "79a873a1146cbdcc385f53e7c14f41aa"},
            {"id": "66611301f6608e9f5815b6bab606351c"},
            {"id": "8e3684c8031804022ff915bb2121e49d"},
        ]

        # Happy case explicitly false (call latest version test)
        response = client.simulate_get("/trials/a?ancestors=false")

        assert response.status == "200 OK"
        assert response.json == [{"id": "8e3684c8031804022ff915bb2121e49d"}]

        # Not a boolean parameter
        response = client.simulate_get("/trials/a?ancestors=42")

        assert response.status == "400 Bad Request"
        assert response.json == {
            "title": "Invalid parameter",
            "description": 'The "ancestors" parameter is invalid. '
            'The value of the parameter must be "true" or "false".',
        }

    def test_trials_by_status(self, client, ephemeral_storage):
        """Tests that trials are returned"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)

        # There exist no trial of the given status in an empty experiment
        response = client.simulate_get("/trials/a?status=completed")

        assert response.status == "200 OK"
        assert response.json == []

        # There exist no trial of the given status while other status are present
        add_trial(storage, experiment=1, id_override="00", status="broken", value=0)

        response = client.simulate_get("/trials/a?status=completed")
        assert response.status == "200 OK"
        assert response.json == []

        # There exist at least one trial of the given status
        add_trial(storage, experiment=1, id_override="01", status="completed", value=1)

        response = client.simulate_get("/trials/a?status=completed")

        assert response.status == "200 OK"
        assert response.json == [{"id": "79a873a1146cbdcc385f53e7c14f41aa"}]

        # Status doesn't exist
        response = client.simulate_get("/trials/a?status=invalid")

        assert response.status == "400 Bad Request"
        error_message = (
            "Invalid status value. "
            "Expected one of "
            "['new', 'reserved', 'suspended', 'completed', 'interrupted', 'broken']"
        )
        assert response.json == {
            "title": "Invalid parameter",
            "description": 'The "status" parameter is invalid. '
            "The value of the parameter must be one of "
            "['new', 'reserved', 'suspended', 'completed', "
            "'interrupted', 'broken']",
        }

    def test_trials_by_from_specific_version_by_status_with_ancestors(
        self, client, ephemeral_storage
    ):
        """Tests that mixing parameters work as intended"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)
        add_experiment(storage, name="b", version=1, _id=2)
        add_experiment(storage, name="a", version=2, _id=3)
        add_experiment(storage, name="a", version=3, _id=4)

        add_trial(storage, experiment=1, id_override="00", value=1, status="completed")
        add_trial(storage, experiment=3, id_override="01", value=2, status="broken")
        add_trial(storage, experiment=3, id_override="02", value=3, status="completed")
        add_trial(storage, experiment=2, id_override="03", value=4, status="completed")

        response = client.simulate_get(
            "/trials/a?ancestors=true&version=2&status=completed"
        )

        assert response.status == "200 OK"
        assert response.json == [
            {"id": "79a873a1146cbdcc385f53e7c14f41aa"},
            {"id": "8e3684c8031804022ff915bb2121e49d"},
        ]


class TestTrialItem:
    """Tests trials/:experiment_name/:trial_id"""

    def test_unknown_experiment(self, client, ephemeral_storage):
        """Tests that an unknown experiment returns a not found error"""
        response = client.simulate_get("/trials/unknown-experiment/a-trial")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Experiment not found",
            "description": 'Experiment "unknown-experiment" does not exist',
        }

    def test_unknown_trial(self, client, ephemeral_storage):
        """Tests that an unknown experiment returns a not found error"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)

        response = client.simulate_get("/trials/a/unknown-trial")

        assert response.status == "404 Not Found"
        assert response.json == {
            "title": "Trial not found",
            "description": 'Trial "unknown-trial" does not exist',
        }

    def test_get_trial(self, client, ephemeral_storage):
        """Tests that an existing trial is returned according to the API specification"""
        storage = ephemeral_storage.storage

        add_experiment(storage, name="a", version=1, _id=1)
        add_trial(storage, experiment=1, id_override="00", status="completed", value=0)
        add_trial(storage, experiment=1, id_override="01", status="completed", value=1)

        response = client.simulate_get("/trials/a/79a873a1146cbdcc385f53e7c14f41aa")

        assert response.status == "200 OK"
        assert response.json == {
            "id": "79a873a1146cbdcc385f53e7c14f41aa",
            "submitTime": "0001-01-01 00:00:00",
            "startTime": "0001-01-01 00:00:10",
            "endTime": "0001-01-02 00:00:00",
            "parameters": {"x": 1},
            "objective": 0.05,
            "statistics": {"a": 10, "b": 5},
        }
