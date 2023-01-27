#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.client`."""
import copy
import json
from importlib import reload

import pytest

import orion.client
import orion.client.cli as cli
import orion.core
from orion.algo.random import Random
from orion.client import get_experiment
from orion.client.experiment import ExperimentClient
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils.exceptions import (
    BranchingEvent,
    NoConfigurationError,
    RaceCondition,
    UnsupportedOperation,
)
from orion.storage.base import setup_storage
from orion.storage.legacy import Legacy
from orion.testing import OrionState

create_experiment = orion.client.create_experiment
workon = orion.client.workon


config = dict(
    name="supernaekei",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "tsirif",
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
    max_broken=5,
    working_dir="",
    algorithm={"random": {"seed": 1}},
    refers=dict(root_id="supernaekei", parent_id=None, adapter=[]),
)


@pytest.fixture()
def user_config():
    """Curate config as a user would provide it"""
    user_config = copy.deepcopy(config)
    user_config.pop("metadata")
    user_config.pop("version")
    user_config.pop("refers")
    user_config.pop("pool_size")
    return user_config


@pytest.fixture()
def data():
    """Return serializable data."""
    return "this is datum"


class TestReportResults:
    """Check functionality and edge cases of `report_results` helper interface."""

    def test_with_no_env(self, monkeypatch, capsys, data):
        """Test without having set the appropriate environmental variable.

        Then: It should print `data` parameter instead to stdout.
        """
        monkeypatch.delenv("ORION_RESULTS_PATH", raising=False)
        reloaded_client = reload(cli)

        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == data + "\n"
        assert err == ""

    def test_with_correct_env(self, monkeypatch, capsys, tmpdir, data):
        """Check that a file with correct data will be written to an existing
        file in a legit path.
        """
        path = str(tmpdir.join("naedw.txt"))
        with open(path, mode="w"):
            pass
        monkeypatch.setenv("ORION_RESULTS_PATH", path)
        reloaded_client = reload(cli)

        assert reloaded_client.IS_ORION_ON is True
        assert reloaded_client.RESULTS_FILENAME == path
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == ""
        assert err == ""

        with open(path) as results_file:
            res = json.load(results_file)
        assert res == data

    def test_with_env_set_but_no_file_exists(self, monkeypatch, tmpdir, data):
        """Check that a Warning will be raised at import time,
        if environmental is set but does not correspond to an existing file.
        """
        path = str(tmpdir.join("naedw.txt"))
        monkeypatch.setenv("ORION_RESULTS_PATH", path)

        with pytest.raises(RuntimeWarning) as exc:
            reload(cli)

        assert "existing file" in str(exc.value)

    def test_call_interface_twice(self, monkeypatch, data):
        """Check that a Warning will be raised at call time,
        if function has already been called once.
        """
        monkeypatch.delenv("ORION_RESULTS_PATH", raising=False)
        reloaded_client = reload(cli)

        reloaded_client.report_results(data)
        with pytest.raises(RuntimeWarning) as exc:
            reloaded_client.report_results(data)

        assert "already reported" in str(exc.value)
        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is True


@pytest.mark.usefixtures("version_XYZ")
class TestCreateExperiment:
    """Test creation of experiment with `client.create_experiment()`"""

    @pytest.mark.usefixtures("orionstate")
    def test_create_experiment_no_storage(self, monkeypatch):
        """Test creation if storage is not configured"""
        name = "oopsie_forgot_a_storage"
        host = orion.core.config.storage.database.host

        with OrionState(storage=orion.core.config.storage.to_dict()) as cfg:
            # Reset the Storage and drop instances so that setup_storage() would fail.
            cfg.cleanup()

            # Make sure storage must be instantiated during `create_experiment()`
            # with pytest.raises(SingletonNotInstantiatedError):
            #    setup_storage()

            experiment = create_experiment(
                name=name, space={"x": "uniform(0, 10)"}, storage=cfg.storage_config
            )

            assert isinstance(experiment._experiment._storage, Legacy)
            assert isinstance(experiment._experiment._storage._db, PickledDB)
            assert experiment._experiment._storage._db.host == host

    def test_create_experiment_new_no_space(self):
        """Test that new experiment needs space"""
        with OrionState() as cfg:
            name = "oopsie_forgot_a_space"
            with pytest.raises(NoConfigurationError) as exc:
                create_experiment(name=name, storage=cfg.storage_config)

            assert f"Experiment {name} does not exist in DB" in str(exc.value)

    def test_create_experiment_bad_storage(self):
        """Test error message if storage is not configured properly"""
        name = "oopsie_bad_storage"
        # Make sure there is no existing storage singleton

        with pytest.raises(NotImplementedError) as exc:
            create_experiment(
                name=name,
                storage={"type": "legacy", "database": {"type": "idontexist"}},
            )

        assert "Could not find implementation of Database, type = 'idontexist'" in str(
            exc.value
        )

    def test_create_experiment_new_default(self):
        """Test creating a new experiment with all defaults"""
        name = "all_default"
        space = {"x": "uniform(0, 10)"}
        with OrionState() as cfg:
            experiment = create_experiment(
                name="all_default", space=space, storage=cfg.storage_config
            )

            assert experiment.name == name
            assert experiment.space.configuration == space

            assert experiment.max_trials == orion.core.config.experiment.max_trials
            assert experiment.max_broken == orion.core.config.experiment.max_broken
            assert experiment.working_dir == orion.core.config.experiment.working_dir
            assert experiment.algorithm
            assert experiment.algorithm.configuration == {"random": {"seed": None}}

    def test_create_experiment_new_full_config(self, user_config):
        """Test creating a new experiment by specifying all attributes."""
        with OrionState() as cfg:
            experiment = create_experiment(**user_config, storage=cfg.storage_config)

            exp_config = experiment.configuration

            assert exp_config["space"] == config["space"]
            assert exp_config["max_trials"] == config["max_trials"]
            assert exp_config["max_broken"] == config["max_broken"]
            assert exp_config["working_dir"] == config["working_dir"]
            assert exp_config["algorithm"] == config["algorithm"]

    def test_create_experiment_hit_no_branch(self, user_config):
        """Test creating an existing experiment by specifying all identical attributes."""
        with OrionState(experiments=[config]) as cfg:
            experiment = create_experiment(**user_config, storage=cfg.storage_config)

            exp_config = experiment.configuration

            assert experiment.name == config["name"]
            assert experiment.version == 1
            assert exp_config["space"] == config["space"]
            assert exp_config["max_trials"] == config["max_trials"]
            assert exp_config["max_broken"] == config["max_broken"]
            assert exp_config["working_dir"] == config["working_dir"]
            assert exp_config["algorithm"] == config["algorithm"]

    def test_create_experiment_hit_no_config(self):
        """Test creating an existing experiment by specifying the name only."""
        with OrionState(experiments=[config]) as cfg:
            experiment = create_experiment(config["name"], storage=cfg.storage_config)

            assert experiment.name == config["name"]
            assert experiment.version == 1
            assert experiment.space.configuration == config["space"]
            assert experiment.algorithm
            assert experiment.algorithm.configuration == config["algorithm"]
            assert experiment.max_trials == config["max_trials"]
            assert experiment.max_broken == config["max_broken"]
            assert experiment.working_dir == config["working_dir"]

    def test_create_experiment_hit_branch(self):
        """Test creating a differing experiment that cause branching."""
        with OrionState(experiments=[config]) as cfg:
            experiment = create_experiment(
                config["name"],
                space={"y": "uniform(0, 10)"},
                branching={"enable": True},
                storage=cfg.storage_config,
            )

            assert experiment.name == config["name"]
            assert experiment.version == 2
            assert experiment.algorithm
            assert experiment.algorithm.configuration == config["algorithm"]
            assert experiment.max_trials == config["max_trials"]
            assert experiment.max_broken == config["max_broken"]
            assert experiment.working_dir == config["working_dir"]

    def test_create_experiment_race_condition(self, monkeypatch):
        """Test that a single race condition is handled seamlessly

        RaceCondition during registration is already handled by `build()`, therefore we will only
        test for race conditions during version update.
        """
        with OrionState(experiments=[config]) as cfg:
            parent = create_experiment(config["name"])
            child = create_experiment(
                config["name"],
                space={"y": "uniform(0, 10)"},
                branching={"enable": True},
                storage=cfg.storage_config,
            )

            def insert_race_condition(self, query):
                is_auto_version_query = query == {
                    "name": config["name"],
                    "refers.parent_id": parent.id,
                }
                if is_auto_version_query:
                    data = [child.configuration]
                # First time the query returns no other child
                elif insert_race_condition.count < 1:
                    data = [parent.configuration]
                else:
                    data = [parent.configuration, child.configuration]

                insert_race_condition.count += int(is_auto_version_query)

                return data

            insert_race_condition.count = 0

            monkeypatch.setattr(
                setup_storage().__class__, "fetch_experiments", insert_race_condition
            )

            experiment = create_experiment(
                config["name"],
                space={"y": "uniform(0, 10)"},
                branching={"enable": True},
            )

            assert insert_race_condition.count == 1
            assert experiment.version == 2
            assert experiment.configuration == child.configuration

    def test_create_experiment_race_condition_broken(self, monkeypatch):
        """Test that two or more race condition leads to raise"""
        with OrionState(experiments=[config]) as cfg:
            parent = create_experiment(config["name"])
            child = create_experiment(
                config["name"],
                space={"y": "uniform(0, 10)"},
                branching={"enable": True},
                storage=cfg.storage_config,
            )

            def insert_race_condition(self, query):
                is_auto_version_query = query == {
                    "name": config["name"],
                    "refers.parent_id": parent.id,
                }
                if is_auto_version_query:
                    data = [child.configuration]
                # The query returns no other child, never!
                else:
                    data = [parent.configuration]

                insert_race_condition.count += int(is_auto_version_query)

                return data

            insert_race_condition.count = 0

            monkeypatch.setattr(
                setup_storage().__class__, "fetch_experiments", insert_race_condition
            )

            with pytest.raises(RaceCondition) as exc:
                create_experiment(
                    config["name"],
                    space={"y": "uniform(0, 10)"},
                    branching={"enable": True},
                )

            assert insert_race_condition.count == 2
            assert "There was a race condition during branching and new version" in str(
                exc.value
            )

    def test_create_experiment_hit_manual_branch(self):
        """Test creating a differing experiment that cause branching."""
        new_space = {"y": "uniform(0, 10)"}
        with OrionState(experiments=[config]) as cfg:
            create_experiment(
                config["name"],
                space=new_space,
                branching={"enable": True},
                storage=cfg.storage_config,
            )

            with pytest.raises(BranchingEvent) as exc:
                create_experiment(
                    config["name"],
                    version=1,
                    space=new_space,
                    branching={"enable": True},
                )

            assert "Configuration is different and generates" in str(exc.value)

    def test_create_experiment_debug_mode(self, tmp_path):
        """Test that EphemeralDB is used in debug mode whatever the storage config given"""

        conf_file = str(tmp_path / "db.pkl")

        experiment = create_experiment(
            config["name"],
            space={"x": "uniform(0, 10)"},
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": conf_file},
            },
        )

        storage = experiment._experiment._storage
        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)

        experiment = create_experiment(
            config["name"],
            space={"x": "uniform(0, 10)"},
            storage={"type": "legacy", "database": {"type": "pickleddb"}},
            debug=True,
        )

        storage = experiment._experiment._storage
        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, EphemeralDB)


class TestWorkon:
    """Test the helper function for sequential API"""

    def test_workon(self):
        """Verify that workon processes properly"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        experiment = workon(foo, space={"x": "uniform(0, 10)"}, max_trials=5)
        assert len(experiment.fetch_trials()) == 5
        assert experiment.name == "loop"
        assert isinstance(experiment._experiment._storage, Legacy)
        assert isinstance(experiment._experiment._storage._db, EphemeralDB)

    def test_workon_algo(self):
        """Verify that algo config is processed properly"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        experiment = workon(
            foo,
            space={"x": "uniform(0, 10)"},
            max_trials=5,
            algorithm={"random": {"seed": 5}},
        )
        assert experiment.algorithm
        algo = experiment.algorithm.unwrapped
        assert isinstance(algo, Random)
        assert algo.seed == 5

    def test_workon_name(self):
        """Verify setting the name with workon"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        experiment = workon(
            foo, space={"x": "uniform(0, 10)"}, max_trials=5, name="voici"
        )

        assert experiment.name == "voici"

    def test_workon_fail(self, monkeypatch):
        """Verify that storage is reverted if workon fails"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        def build_fail(*args, **kwargs):
            raise RuntimeError("You shall not build!")

        monkeypatch.setattr("orion.core.io.experiment_builder.build", build_fail)

        # Flush storage singleton

        with pytest.raises(RuntimeError) as exc:
            experiment = workon(
                foo, space={"x": "uniform(0, 10)"}, max_trials=5, name="voici"
            )

        assert exc.match("You shall not build!")

        # Now test with a prior storage
        with OrionState(
            storage={"type": "legacy", "database": {"type": "EphemeralDB"}}
        ) as cfg:
            storage = cfg.storage

            with pytest.raises(RuntimeError) as exc:
                workon(foo, space={"x": "uniform(0, 10)"}, max_trials=5, name="voici")

            assert exc.match("You shall not build!")

    def test_workon_twice(self):
        """Verify setting the each experiment has its own storage"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        experiment = workon(
            foo, space={"x": "uniform(0, 10)"}, max_trials=5, name="voici"
        )

        assert experiment.name == "voici"
        assert len(experiment.fetch_trials()) == 5

        experiment2 = workon(
            foo, space={"x": "uniform(0, 10)"}, max_trials=1, name="voici"
        )

        assert experiment2.name == "voici"
        assert len(experiment2.fetch_trials()) == 1

    def test_workon_with_parallel_backend(self):
        """Test there is no impact of joblib parallel for workon function"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        import joblib

        with joblib.parallel_backend("loky"):
            experiment = workon(
                foo, space={"x": "uniform(0, 10)"}, max_trials=5, name="voici"
            )

        assert experiment.name == "voici"
        assert len(experiment.fetch_trials()) == 5

        with joblib.parallel_backend("loky", n_jobs=-1):
            experiment = workon(
                foo, space={"x": "uniform(0, 10)"}, max_trials=3, name="voici"
            )

        assert experiment.name == "voici"
        assert len(experiment.fetch_trials()) == 3


class TestGetExperiment:
    """Test :meth:`orion.client.get_experiment`"""

    @pytest.mark.usefixtures("mock_database")
    def test_experiment_do_not_exist(self):
        """Tests that an error is returned when the experiment doesn't exist"""
        with pytest.raises(NoConfigurationError) as exception:
            get_experiment("a")
        assert (
            "No experiment with given name 'a' and version '*' inside database, "
            "no view can be created." == str(exception.value)
        )

    def test_experiment_exist(self, mock_database):
        """
        Tests that an instance of :class:`orion.client.experiment.ExperimentClient` is
        returned representing the latest version when none is given.
        """
        experiment = create_experiment(
            "a", space={"x": "uniform(0, 10)"}, storage=mock_database.storage
        )

        experiment = get_experiment("a", storage=mock_database.storage)

        assert experiment
        assert isinstance(experiment, ExperimentClient)
        assert experiment.mode == "r"

    def test_version_do_not_exist(self, caplog, mock_database):
        """Tests that a warning is printed when the experiment exist but the version doesn't"""
        create_experiment(
            "a", space={"x": "uniform(0, 10)"}, storage=mock_database.storage
        )

        experiment = get_experiment("a", 2, storage=mock_database.storage)

        assert experiment.version == 1
        assert (
            "Version 2 was specified but most recent version is only 1. Using 1."
            in caplog.text
        )

    def test_read_write_mode(self, mock_database):
        """Tests that experiment can be created in write mode"""
        experiment = create_experiment(
            "a", space={"x": "uniform(0, 10)"}, storage=mock_database.storage
        )
        assert experiment.mode == "x"

        experiment = get_experiment("a", 2, mode="r", storage=mock_database.storage)
        assert experiment.mode == "r"

        with pytest.raises(UnsupportedOperation) as exc:
            experiment.insert({"x": 0})

        assert exc.match("ExperimentClient must have write rights to execute `insert()")

        experiment = get_experiment("a", 2, mode="w", storage=mock_database.storage)
        assert experiment.mode == "w"

        trial = experiment.insert({"x": 0})

        with pytest.raises(UnsupportedOperation) as exc:
            experiment.reserve(trial)

        assert exc.match(
            "ExperimentClient must have execution rights to execute `reserve()"
        )
