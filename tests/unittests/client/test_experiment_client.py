#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.experiment`."""
import copy
import datetime
import logging

import joblib
import pandas.testing
from orion.client.experiment import AlreadyReleased
import pytest

import orion.core
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    SampleTimeout,
)
from orion.core.worker.trial import Trial
from orion.executor.base import Executor
from orion.executor.joblib_backend import Joblib
from orion.storage.base import get_storage
from orion.testing import create_experiment, mock_space_iterate

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
    max_trials=10,
    max_broken=5,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
    refers=dict(root_id="supernaekei", parent_id=None, adapter=[]),
)


base_trial = {
    "experiment": 0,
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def compare_trials(trials_a, trials_b):
    """Compare two trials by using their configuration"""

    def to_dict(trial):
        return trial.to_dict()

    assert list(map(to_dict, trials_a)) == list(map(to_dict, trials_b))


def compare_without_heartbeat(trial_a, trial_b):
    """Compare trials configuration ommiting heartbeat"""
    trial_a_dict = trial_a.to_dict()
    trial_b_dict = trial_b.to_dict()
    trial_a_dict.pop("heartbeat")
    trial_b_dict.pop("heartbeat")
    assert trial_a_dict == trial_b_dict


def test_plot_is_defined():
    """Tests plot() method is defined"""
    with create_experiment(config, base_trial) as (_, _, client):
        assert client.plot()


def test_experiment_fetch_trials():
    """Test compliance of client and experiment `fetch_trials()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        assert len(experiment.fetch_trials()) == 5
        compare_trials(experiment.fetch_trials(), client.fetch_trials())


def test_experiment_get_trial():
    """Test compliance of client and experiment `get_trial()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        assert experiment.get_trial(uid=0) == client.get_trial(uid=0)


def test_experiment_fetch_trials_by_status():
    """Test compliance of client and experiment `fetch_trials_by_status()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_trials_by_status("completed"),
            client.fetch_trials_by_status("completed"),
        )


def test_experiment_fetch_pending_trials():
    """Test compliance of client and experiment `fetch_pending_trials()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        compare_trials(experiment.fetch_pending_trials(), client.fetch_pending_trials())


def test_experiment_fetch_non_completed_trials():
    """Test compliance of client and experiment `fetch_noncompleted_trials()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_noncompleted_trials(), client.fetch_noncompleted_trials()
        )


def test_experiment_to_pandas():
    """Test compliance of client and experiment `to_pandas()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        pandas.testing.assert_frame_equal(experiment.to_pandas(), client.to_pandas())


@pytest.mark.usefixtures("version_XYZ")
class TestInsert:
    """Tests for ExperimentClient.insert"""

    def test_insert_params_wo_results(self):
        """Test insertion without results without reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100))
            assert trial.status == "interrupted"
            assert trial.params["x"] == 100
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_params_with_results(self):
        """Test insertion with results without reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            timestamp = datetime.datetime.utcnow()
            trial = client.insert(
                dict(x=100), [dict(name="objective", type="objective", value=101)]
            )
            assert trial.status == "completed"
            assert trial.params["x"] == 100
            assert trial.objective.value == 101
            assert trial.end_time >= timestamp
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))
            assert client.get_trial(uid=trial.id).objective.value == 101

            assert client._pacemakers == {}

    def test_insert_params_with_results_and_reserve(self):
        """Test insertion with results and reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100),
                    [dict(name="objective", type="objective", value=101)],
                    reserve=True,
                )

            assert "Cannot observe a trial and reserve it" in str(exc.value)

    def test_insert_existing_params(self, monkeypatch):
        """Test that duplicated trials cannot be saved in storage"""
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError) as exc:
                client.insert(dict(x=1))

            assert (
                "A trial with params {'x': 1} already exist for experiment supernaekei-v1"
                == str(exc.value)
            )

            assert client._pacemakers == {}

    def test_insert_partial_params(self):
        """Test that trial with missing dimension that has a default value can be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10, default_value=5)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})
        with create_experiment(config_with_default, trial_with_default) as (
            _,
            experiment,
            client,
        ):
            trial = client.insert(dict(x=100))

            assert trial.status == "interrupted"
            assert trial.params["x"] == 100
            assert trial.params["y"] == 5
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_partial_params_missing(self):
        """Test that trial with missing dimension cannot be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})
        with create_experiment(config_with_default, trial_with_default) as (
            _,
            _,
            client,
        ):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x=1))

            assert (
                "Dimension y not specified and does not have a default value."
                == str(exc.value)
            )

    def test_insert_params_and_reserve(self):
        """Test that new trial is reserved properly with `reserve=True`"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100), reserve=True)
            assert trial.status == "reserved"
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_insert_params_fails_not_reserved(self, monkeypatch):
        """Test that failed insertion because of duplicated trials will not reserve the original
        trial
        """
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError):
                client.insert(dict(x=1), reserve=True)

            assert client._pacemakers == {}

    def test_insert_bad_params(self):
        """Test that bad params cannot be registered in storage"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x="bad bad bad"))

            assert (
                "Dimension x value bad bad bad is outside of prior uniform(0, 200)"
                == str(exc.value)
            )
            assert client._pacemakers == {}

    def test_insert_params_bad_results(self):
        """Test that results with from format cannot be saved (trial is registered anyhow)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100), [dict(name="objective", type="bad bad bad", value=0)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.usefixtures("version_XYZ")
class TestReserve:
    """Tests for ExperimentClient.reserve"""

    def test_reserve(self):
        """Test reservation of registered trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            assert trial.status == "reserved"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_dont_exist(self):
        """Verify that unregistered trials cannot be reserved."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.reserve(trial)

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_reserve_reserved_locally(self, caplog):
        """Verify that a trial cannot be reserved twice locally (warning, no exception)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            with caplog.at_level(logging.WARNING):
                client.reserve(trial)

            assert (
                "Trial {} is already reserved.".format(trial.id)
                == caplog.records[-1].message
            )

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_reserved_remotely(self):
        """Verify that a trial cannot be reserved if already reserved by another process"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.status == "interrupted"
            client.reserve(trial)
            remote_pacemaker = client._pacemakers.pop(trial.id)
            assert experiment.get_trial(trial).status == "reserved"

            trial = Trial(**cfg.trials[1])
            assert trial.status == "interrupted"
            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert "Could not reserve trial {}.".format(trial.id) == str(exc.value)

            assert trial.status == "interrupted"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers == {}
            remote_pacemaker.stop()

    def test_reserve_race_condition(self):
        """Verify that race conditions during `reserve` is detected and raises a comprehensible
        error
        """
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[0]["_id"])
            experiment.set_trial_status(trial, "reserved")
            trial.status = "new"  # Let's pretend it is still available

            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert "Could not reserve trial {}.".format(trial.id) == str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.usefixtures("version_XYZ")
class TestRelease:
    """Tests for ExperimentClient.release"""

    def test_release(self):
        """Test releasing (to interrupted)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial)
            assert trial.status == "interrupted"
            assert experiment.get_trial(trial).status == "interrupted"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_status(self):
        """Test releasing with a specific status"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial, "broken")
            assert trial.status == "broken"
            assert experiment.get_trial(trial).status == "broken"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_invalid_status(self):
        """Test releasing with a specific status"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            with pytest.raises(ValueError) as exc:
                client.release(trial, "mouf mouf")

            assert exc.match("Given status `mouf mouf` not one of")

    def test_release_dont_exist(self, monkeypatch):
        """Verify that unregistered trials cannot be released"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[1]["params"])

            def do_nada(trial, **kwargs):
                """Don't do anything"""
                return None

            monkeypatch.setattr(client, "_release_reservation", do_nada)

            with pytest.raises(ValueError) as exc:
                client.release(trial)

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_release_race_condition(self):
        """Verify that race conditions during `release` is detected and raises a comprehensible
        error
        """
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            # Woops! Trial got failed over from another process.
            experiment.set_trial_status(trial, "interrupted")
            trial.status = "reserved"  # Let's pretend we don't know.

            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert "Reservation for trial {} has been lost before release.".format(
                trial.id
            ) in str(exc.value)
            assert client._pacemakers == {}
            assert not pacemaker.is_alive()

    def test_release_unreserved(self):
        """Verify that unreserved trials cannot be released"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            with pytest.raises(AlreadyReleased) as exc:
                client.release(trial)

            assert "Trial {} was already released locally.".format(trial.id) == str(
                exc.value
            )

            assert client._pacemakers == {}

    def test_release_already_released_but_incorrectly(self):
        """Verify that incorrectly released trials have its pacemaker stopped properly"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            assert trial.status == "reserved"
            experiment.set_trial_status(trial, "interrupted")
            assert trial.status == "interrupted"

            with pytest.raises(AlreadyReleased) as exc:
                client.release(trial)

            assert "Trial {} was already released locally.".format(trial.id) == str(
                exc.value
            )

            assert client._pacemakers == {}
            assert not pacemaker.is_alive()


@pytest.mark.usefixtures("version_XYZ")
class TestClose:
    """Test close method of the client"""

    def test_close_empty(self):
        """Test client can close when no trial is reserved"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            client.close()

    def test_close_with_reserved(self):
        """Test client cannot be closed if trials are reserved."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                with pytest.raises(RuntimeError) as exc:
                    client.close()

                assert "There is still reserved trials" in str(exc.value)


@pytest.mark.usefixtures("version_XYZ")
class TestBroken:
    """Test handling of broken trials"""

    def test_broken_trial(self):
        """Test that broken trials are detected"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(RuntimeError):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    raise RuntimeError("Dummy failure!")

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "broken"

    def test_interrupted_trial(self):
        """Test that interrupted trials are not set to broken"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    raise KeyboardInterrupt

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "interrupted"

    def test_completed_then_interrupted_trial(self):
        """Test that interrupted trials are not set to broken"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    assert trial.results == []
                    assert get_storage().get_trial(trial).objective is None
                    client.observe(
                        trial, [dict(name="objective", type="objective", value=101)]
                    )
                    assert get_storage().get_trial(trial).objective.value == 101
                    assert trial.status == "completed"
                    raise KeyboardInterrupt

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "completed"


@pytest.mark.usefixtures("version_XYZ")
class TestSuggest:
    """Tests for ExperimentClient.suggest"""

    def test_suggest(self, monkeypatch):
        """Verify that suggest reserved availabe trials."""
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == 1

            assert len(experiment.fetch_trials()) == 5
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_suggest_new(self):
        """Verify that suggest can create, register and reserved new trials."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            for _ in range(3):
                trial = client.suggest()
                assert trial.status == "reserved"
                assert len(experiment.fetch_trials()) == 5
                assert client._pacemakers[trial.id].is_alive()
                client._pacemakers[trial.id].stop()

            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == 57.57
            assert len(experiment.fetch_trials()) > 5

            assert client._pacemakers[trial.id].is_alive()
            for trial_id in list(client._pacemakers.keys()):
                client._pacemakers.pop(trial_id).stop()

    def test_suggest_race_condition(self, monkeypatch):
        """Verify that race conditions to register new trials is handled"""
        mock_space_iterate(monkeypatch)
        new_value = 50.0

        # algo will suggest once an already existing trial
        def amnesia(num=1):
            """Suggest a new value and then always suggest the same"""
            if amnesia.count == 0:
                value = [0]
            else:
                value = [new_value]

            amnesia.count += 1

            return [value]

        amnesia.count = 0

        with create_experiment(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(experiment.algorithms, "suggest", amnesia)

            assert len(experiment.fetch_trials()) == 1

            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == new_value
            assert amnesia.count == 2

            assert len(experiment.fetch_trials()) == 2
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_suggest_algo_opt_out(self, monkeypatch):
        """Verify that None is returned when algo cannot sample new trials (opting opt)"""

        def opt_out(num=1):
            """Never suggest a new trial"""
            return None

        monkeypatch.setattr(orion.core.config.worker, "max_idle_time", -1)

        with create_experiment(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(experiment.algorithms, "suggest", opt_out)

            assert len(experiment.fetch_trials()) == 1

            with pytest.raises(SampleTimeout):
                client.suggest()

    def test_suggest_is_done(self):
        """Verify that completed experiments cannot suggest new trials"""
        with create_experiment(config, base_trial, statuses=["completed"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            with pytest.raises(CompletedExperiment):
                client.suggest()

    def test_suggest_is_done_context_manager(self):
        """Verify that context manager handles None"""
        with create_experiment(config, base_trial, statuses=["completed"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            with pytest.raises(CompletedExperiment):
                client.suggest()

    def test_suggest_is_broken(self):
        """Verify that broken experiments cannot suggest new trials"""
        with create_experiment(config, base_trial, statuses=["broken"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

    def test_suggest_is_done_race_condition(self, monkeypatch):
        """Verify that inability to suggest because is_done becomes True during produce() is
        handled.
        """
        with create_experiment(config, base_trial, statuses=["completed"] * 5) as (
            cfg,
            experiment,
            client,
        ):

            def is_done(self):
                """Experiment is done"""
                return True

            def set_is_done(pool_size):
                """Set is_done while algo is trying to suggest"""
                monkeypatch.setattr(experiment.__class__, "is_done", property(is_done))

            monkeypatch.setattr(client._producer, "produce", set_is_done)

            assert len(experiment.fetch_trials()) == 5
            assert not client.is_done

            with pytest.raises(CompletedExperiment):
                client.suggest()

            assert len(experiment.fetch_trials()) == 5
            assert client.is_done

    def test_suggest_is_broken_race_condition(self, monkeypatch):
        """Verify that experiments that gets broken during local algo.suggest gets properly
        handled
        """
        with create_experiment(config, base_trial, statuses=["broken"] * 1) as (
            cfg,
            experiment,
            client,
        ):

            def is_broken(self):
                """Experiment is broken"""
                return True

            def set_is_broken(pool_size):
                """Set is_broken while algo is trying to suggest"""
                monkeypatch.setattr(
                    experiment.__class__, "is_broken", property(is_broken)
                )

            monkeypatch.setattr(client._producer, "produce", set_is_broken)

            assert len(experiment.fetch_trials()) == 1
            assert not client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

            assert len(experiment.fetch_trials()) == 1
            assert client.is_broken

    def test_suggest_hierarchical_space(self):
        """Verify that suggest returns trial with proper hierarchical parameter."""
        exp_config = copy.deepcopy(config)
        exp_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)", "z": "choices(['voici', 'voila', 2])"},
        }
        with create_experiment(
            exp_config=exp_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert len(trial.params) == 2
            assert "x" in trial.params["a"]
            assert "y" in trial.params["b"]
            assert "z" in trial.params["b"]

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()


@pytest.mark.usefixtures("version_XYZ")
class TestObserve:
    """Tests for ExperimentClient.observe"""

    def test_observe(self):
        """Verify that `observe()` will update the storage"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.results == []
            client.reserve(trial)
            assert get_storage().get_trial(trial).objective is None
            client.observe(trial, [dict(name="objective", type="objective", value=101)])
            assert get_storage().get_trial(trial).objective.value == 101

    def test_observe_unreserved(self):
        """Verify that `observe()` will fail on non-reserved trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Trial {} had no pacemakers. Was it reserved properly?".format(
                trial.id
            ) == str(exc.value)

    def test_observe_dont_exist(self):
        """Verify that `observe()` will fail on non-registered trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_observe_bad_results(self):
        """Verify that bad results type is detected and ValueError is raised"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            client.reserve(trial)
            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="bad bad bad", value=101)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_observe_race_condition(self):
        """Verify that race condition during `observe()` is detected and raised"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            experiment.set_trial_status(trial, "interrupted")
            trial.status = "reserved"  # Let's pretend it is still reserved

            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Reservation for trial {} has been lost.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_observe_under_with(self):
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                assert trial.status == "reserved"
                assert trial.results == []
                assert get_storage().get_trial(trial).objective is None
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )
                assert get_storage().get_trial(trial).objective.value == 101
                assert trial.status == "completed"

            assert trial.status == "completed"  # Still completed after __exit__


def foo_1(x):
    return [dict(name="result", type="objective", value=x * 2)]


def foo_2(x, y):
    return [dict(name="result", type="objective", value=x * 2 + y)]


default_y = 2
default_z = "voila"

def foo_test_workon_hierarchical_partial_with_override(a, b):
    assert b["y"] != default_y
    assert b["z"] == default_z
    return [dict(name="result", type="objective", value=a["x"] * 2 + b["y"])]


def foo_error(x):
    raise RuntimeError()


def foo_maybe_error(x):
    foo_maybe_error.count += 1
    if foo_maybe_error.count < 5:
        raise RuntimeError()

    return [dict(name="result", type="objective", value=x * 2)]

foo_maybe_error.count = 0


def foo_trial_args(x, my_trial_arg_name):
    assert isinstance(my_trial_arg_name, Trial)
    assert my_trial_arg_name.params["x"] == x
    return [dict(name="result", type="objective", value=x * 2)]


def foo_on_error(x, q):
    if not q.empty:
        raise q.get()()

    return [dict(name="result", type="objective", value=x * 2)]


def foo_reraise(x):
    raise NotImplementedError("Do not ignore this!")


@pytest.mark.usefixtures("version_XYZ")
class TestWorkon:
    """Tests for ExperimentClient.workon"""

    def test_workon(self):
        """Verify that workon processes properly"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_1, max_trials=5)
            assert len(experiment.fetch_trials_by_status("completed")) == 5
            assert client._pacemakers == {}

    def test_workon_partial(self):
        """Verify that partial is properly passed to the function"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_2, max_trials=10, y=2)
            assert len(experiment.fetch_trials()) == 10
            assert client._pacemakers == {}

    def test_workon_partial_with_override(self):
        """Verify that partial is overriden by trial.params"""


        ext_config = copy.deepcopy(config)
        ext_config["space"]["y"] = "uniform(0, 10)"

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            default_y = 2
            assert len(experiment.fetch_trials()) == 0
            client.workon(foo_2, max_trials=1, y=default_y)
            assert len(experiment.fetch_trials_by_status("completed")) == 1
            assert experiment.fetch_trials()[0].params["y"] != 2

    def test_workon_hierarchical_partial_with_override(self):
        """Verify that hierarchical partial is overriden by trial.params"""
        default_y = 2
        default_z = "voila"

        ext_config = copy.deepcopy(config)
        ext_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)"},
        }

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            assert len(experiment.fetch_trials()) == 0
            client.workon(foo_test_workon_hierarchical_partial_with_override, max_trials=5, b={"y": default_y, "z": default_z})
            assert len(experiment.fetch_trials_by_status("completed")) == 5
            params = experiment.fetch_trials()[0].params
            assert len(params)
            assert "x" in params["a"]
            assert "y" in params["b"]

    def test_workon_max_trials(self):
        """Verify that workon stop when reaching max_trials"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            client.workon(foo_1, max_trials=MAX_TRIALS)
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS

    def test_workon_max_trials_resumed(self):
        """Verify that workon stop when reaching max_trials after resuming"""

        with create_experiment(
            config, base_trial, statuses=["completed", "completed"]
        ) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            assert len(experiment.fetch_trials_by_status("completed")) == 2
            client.workon(foo_1, max_trials=MAX_TRIALS)
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS

    def test_workon_max_trials_per_worker(self):
        """Verify that workon stop when reaching max_trials_per_worker"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=MAX_TRIALS - 1
            )
            assert executed == MAX_TRIALS - 1
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS - 1

    def test_workon_max_trials_per_worker_resumed(self):
        """Verify that workon stop when reaching max_trials_per_worker after resuming"""

        n_completed = 2
        statuses = ["completed"] * n_completed + ["new"]
        n_trials = len(statuses)

        with create_experiment(config, base_trial, statuses=statuses) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 9
            assert client.max_trials > MAX_TRIALS
            assert len(experiment.fetch_trials_by_status("completed")) == n_completed
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=2
            )
            assert executed == 2
            assert (
                len(experiment.fetch_trials_by_status("completed")) == 2 + n_completed
            )
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=3
            )
            assert executed == 3
            assert (
                len(experiment.fetch_trials_by_status("completed"))
                == 3 + 2 + n_completed
            )

    def test_workon_exp_max_broken_before_worker_max_broken(self):
        """Verify that workon stop when reaching exp.max_broken"""

        MAX_TRIALS = 5
        MAX_BROKEN = 20
        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN // 2

        with create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(BrokenExperiment):
                client.workon(foo_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN // 2
            assert n_trials - n_broken_trials < MAX_TRIALS

    def test_workon_max_broken_all_broken(self):
        """Verify that workon stop when reaching worker's max_broken"""

        MAX_TRIALS = 5
        MAX_BROKEN = 10

        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN * 2

        with create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(BrokenExperiment):
                client.workon(foo_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN
            assert n_trials - n_broken_trials < MAX_TRIALS

    def test_workon_max_trials_before_max_broken(self):
        """Verify that workon stop when reaching max_trials before max_broken"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            MAX_TRIALS = 5
            MAX_BROKEN = 10
            assert client.max_trials > MAX_TRIALS
            client.workon(foo_maybe_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials < MAX_BROKEN
            assert n_trials - n_broken_trials == MAX_TRIALS

    def test_workon_trial_arg(self):
        """Verify that workon pass trial when trial_arg is defined"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_trial_args, max_trials=5, trial_arg="my_trial_arg_name")
            assert len(experiment.fetch_trials()) == 5

    def test_workon_on_error_ignore(self):
        """Verify that workon on_error callback ignores some errors correctly"""

        def on_error(client, trial, error, worker_broken_trials):
            assert on_error.counter == worker_broken_trials
            if isinstance(error, (IndexError, IOError, AttributeError)):
                client.release(trial, "cancelled")
                return False

            on_error.counter += 1
            return True

        on_error.counter = 0

        errors = [
            IndexError,
            ValueError,
            IOError,
            NotImplementedError,
            AttributeError,
            ImportError,
        ]
        MAX_TRIALS = 5
        MAX_BROKEN = len(errors) + 1

        def make_error_queue():
            from multiprocessing import Manager
            m = Manager()
            q = m.Queue()
            for e in errors:
                q.put(e)

            return m, q

        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN * 2

        manager, errors = make_error_queue()

        with manager, create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            client.workon(foo_on_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN, q=errors)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN - 1
            assert n_trials - n_broken_trials == MAX_TRIALS

    def test_workon_on_error_raise(self):
        """Verify that workon on_error callback can raise and stop iteration"""

        def on_error(client, trial, error, worker_broken_trials):
            raise error

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(NotImplementedError) as exc:
                client.workon(foo_reraise, max_trials=5, max_broken=5, on_error=on_error)

            print(exc)
            assert exc.match("Do not ignore this!")

    def test_parallel_workers(self, monkeypatch):
        """Test parallel execution with joblib"""

        def optimize(*args, **kwargs):
            return 1

        with create_experiment(exp_config=config, trial_config={}, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(client, "_optimize", optimize)
            with client.tmp_executor("joblib", n_workers=5, backend="threading"):
                trials = client.workon(foo_1, max_trials=5, n_workers=2)
            assert trials == 2

            with client.tmp_executor("joblib", n_workers=5, backend="threading"):
                trials = client.workon(foo_1, max_trials=5, n_workers=3)
            assert trials == 3

            executor = Joblib(n_workers=5, backend="threading")
            client.executor = executor
            trials = client.workon(foo_1, max_trials=5, n_workers=4)
            assert trials == 4
