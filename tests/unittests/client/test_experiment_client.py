#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.client.experiment`."""
import copy
import datetime
import logging
import os
import time

import pandas.testing
import pytest

import orion.core
from orion.client.experiment import reserve_trial
from orion.core.io.database import DuplicateKeyError
from orion.core.utils import format_trials
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    ReservationRaceCondition,
    WaitingForTrials,
)
from orion.core.worker.trial import AlreadyReleased, Trial
from orion.executor.base import ExecutorClosed, executor_factory
from orion.testing import create_experiment, create_rest_experiment, mock_space_iterate

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

factories = [
    create_experiment,
]


def is_running_tox_service():
    return os.getenv("TOX_ENV_NAME") == "service"


if is_running_tox_service():
    factories = [create_rest_experiment]


def is_rest(factory):
    return factory is create_rest_experiment


def subset_match(a, b):
    keys = list(a.keys())
    if len(keys) > len(list(b.keys())):
        keys = list(b.keys())

    for k in keys:
        assert a[k] == b[k]

    return True


def compare_trials(trials_a, trials_b, factory):
    """Compare two trials by using their configuration"""

    def to_dict(trial):
        return trial.to_dict()

    if is_rest(factory):
        for ta, tb in zip(trials_a, trials_b):
            assert subset_match(to_dict(ta), to_dict(tb))

        return

    assert list(map(to_dict, trials_a)) == list(map(to_dict, trials_b))


def compare_without_heartbeat(trial_a, trial_b):
    """Compare trials configuration omitting heartbeat"""
    trial_a_dict = trial_a.to_dict()
    trial_b_dict = trial_b.to_dict()

    trial_a_dict.pop("heartbeat", None)
    trial_b_dict.pop("heartbeat", None)
    assert trial_a_dict == trial_b_dict


@pytest.mark.parametrize("factory", factories)
def test_plot_is_defined(factory):
    """Tests plot() method is defined"""

    if is_rest(factory):
        pytest.skip("Not implemented")
        return

    with factory(config, base_trial) as (_, _, client):
        assert client.plot()


@pytest.mark.parametrize("factory", factories)
def test_experiment_fetch_trials(factory):
    """Test compliance of client and experiment `fetch_trials()`"""
    with factory(config, base_trial) as (cfg, experiment, client):

        assert len(experiment.fetch_trials()) == 5
        compare_trials(experiment.fetch_trials(), client.fetch_trials(), factory)


@pytest.mark.parametrize("factory", factories)
def test_experiment_get_trial(factory):
    """Test compliance of client and experiment `get_trial()`"""
    with factory(config, base_trial) as (cfg, experiment, client):
        assert experiment.get_trial(uid=0) == client.get_trial(uid=0)


@pytest.mark.parametrize("factory", factories)
def test_experiment_fetch_trials_by_status(factory):
    """Test compliance of client and experiment `fetch_trials_by_status()`"""
    with factory(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_trials_by_status("completed"),
            client.fetch_trials_by_status("completed"),
            factory,
        )


@pytest.mark.parametrize("factory", factories)
def test_experiment_fetch_pending_trials(factory):
    """Test compliance of client and experiment `fetch_pending_trials()`"""
    with factory(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_pending_trials(), client.fetch_pending_trials(), factory
        )


@pytest.mark.parametrize("factory", factories)
def test_experiment_fetch_non_completed_trials(factory):
    """Test compliance of client and experiment `fetch_noncompleted_trials()`"""
    with factory(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_noncompleted_trials(),
            client.fetch_noncompleted_trials(),
            factory,
        )


@pytest.mark.parametrize("factory", factories)
def test_experiment_to_pandas(factory):
    """Test compliance of client and experiment `to_pandas()`"""
    if is_rest(factory):
        pytest.skip("Not implemented")
        return

    with factory(config, base_trial) as (cfg, experiment, client):
        pandas.testing.assert_frame_equal(experiment.to_pandas(), client.to_pandas())


@pytest.mark.parametrize("factory", factories)
class TestReservationFct:
    def test_no_sample(self, monkeypatch, factory):
        """Test that WaitingForTrials is raised when exp unable to reserve trials."""

        if is_rest(factory):
            pytest.skip("cannot patch producer, does not exist for REST API")
            return

        with factory(config, base_trial, ["reserved"]) as (
            cfg,
            experiment,
            client,
        ):

            N_TRIALS = 5

            def do_nothing(pool_size):
                # Another worker generates a trial meanwhile
                n_trials_so_far = len(client.fetch_trials())
                if n_trials_so_far < N_TRIALS:
                    client.insert(
                        dict(x=n_trials_so_far),
                        results=[
                            {"name": "objective", "type": "objective", "value": 1}
                        ],
                    )
                return 0

            monkeypatch.setattr(client._producer, "produce", do_nothing)

            start = time.time()

            with pytest.raises(WaitingForTrials) as exc:
                reserve_trial(experiment, client._producer, pool_size=1)

    def test_stops_if_exp_done(self, monkeypatch, factory):
        """Test that reservation attempt is stopped when experiment is done."""
        if is_rest(factory):
            pytest.skip("cannot patch producer, does not exist for REST API")
            return

        with factory(config, base_trial, ["reserved"]) as (
            cfg,
            experiment,
            client,
        ):

            # Make sure first it produces properly
            n_trials_before_reserve = len(client.fetch_trials())
            assert not client.is_done

            reserve_trial(experiment, client._producer, pool_size=1)

            assert not client.is_done
            assert len(client.fetch_trials()) == n_trials_before_reserve + 1

            # Then make sure the producer raises properly
            def cant_produce(pool_size):
                raise RuntimeError("I should not be called")

            monkeypatch.setattr(client._producer, "produce", cant_produce)

            with pytest.raises(RuntimeError, match="I should not be called"):
                reserve_trial(experiment, client._producer, pool_size=1)

            # Now make sure the producer is not called and no additional trials are generated
            def make_exp_is_done(reserve):
                # Another worker generates a trial meanwhile
                monkeypatch.setattr(experiment, "max_trials", 0)
                assert experiment.is_done
                return None

            monkeypatch.setattr(type(experiment), "reserve_trial", make_exp_is_done)

            n_trials_before_reserve = len(client.fetch_trials())
            assert not client.is_done

            with pytest.raises(CompletedExperiment):
                reserve_trial(experiment, client._producer, pool_size=1)

            assert client.is_done
            assert len(client.fetch_trials()) == n_trials_before_reserve


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestInsert:
    """Tests for ExperimentClient.insert"""

    def test_insert_params_wo_results(self, factory):
        """Test insertion without results without reservation"""
        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100))

            ref_trial = experiment.get_trial(uid=trial.id)
            assert ref_trial.status == "new"

            if not is_rest(factory):
                compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}
            assert trial.params["x"] == 100
            assert trial.id in {trial.id for trial in experiment.fetch_trials()}

    def test_insert_params_with_results(self, factory):
        """Test insertion with results without reservation"""
        with factory(config, base_trial) as (cfg, experiment, client):
            timestamp = datetime.datetime.utcnow()
            trial = client.insert(
                dict(x=100), [dict(name="objective", type="objective", value=101)]
            )

            ref_trial = experiment.get_trial(uid=trial.id)
            assert ref_trial.status == "completed"
            assert trial.params["x"] == 100

            if not is_rest(factory):
                assert trial.objective.value == 101
                assert trial.end_time >= timestamp
                compare_without_heartbeat(trial, client.get_trial(uid=trial.id))
                assert client.get_trial(uid=trial.id).objective.value == 101

            assert client._pacemakers == {}
            assert trial.id in {trial.id for trial in experiment.fetch_trials()}

    def test_insert_params_with_results_and_reserve(self, factory):
        """Test insertion with results and reservation"""

        if is_rest(factory):
            pytest.skip("Cannot reserve inserted trials with REST API")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100),
                    [dict(name="objective", type="objective", value=101)],
                    reserve=True,
                )

            assert "Cannot observe a trial and reserve it" in str(exc.value)

    def test_insert_existing_params(self, monkeypatch, factory):
        """Test that duplicated trials cannot be saved in storage"""
        mock_space_iterate(monkeypatch)

        with factory(config, base_trial) as (cfg, experiment, client):

            with pytest.raises(DuplicateKeyError) as exc:
                client.insert(dict(x=1))

            assert (
                "A trial with params {'x': 1} already exist for experiment supernaekei-v1"
                == str(exc.value)
            )

            assert client._pacemakers == {}

    def test_insert_partial_params(self, factory):
        """Test that trial with missing dimension that has a default value can be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10, default_value=5)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})

        with factory(config_with_default, trial_with_default) as (
            _,
            experiment,
            client,
        ):
            trial = client.insert(dict(x=100))

            ref_trial = experiment.get_trial(uid=trial.id)
            assert ref_trial.status == "new"

            assert trial.params["x"] == 100
            assert trial.params["y"] == 5

            assert trial.id in {trial.id for trial in experiment.fetch_trials()}
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))
            assert client._pacemakers == {}

    def test_insert_partial_params_missing(self, factory):
        """Test that trial with missing dimension cannot be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})
        with factory(config_with_default, trial_with_default) as (
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

    def test_insert_params_and_reserve(self, factory):
        """Test that new trial is reserved properly with `reserve=True`"""
        if is_rest(factory):
            pytest.skip("API REST does not reserve inserted trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100), reserve=True)

            ref_trial = experiment.get_trial(uid=trial.id)
            assert ref_trial.status == "reserved"

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_insert_params_fails_not_reserved(self, factory, monkeypatch):
        """Test that failed insertion because of duplicated trials will not reserve the original
        trial
        """
        mock_space_iterate(monkeypatch)
        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError):
                client.insert(dict(x=1), reserve=True)

            assert client._pacemakers == {}

    def test_insert_bad_params(self, factory):
        """Test that bad params cannot be registered in storage"""
        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x="bad bad bad"))

            assert "Parameters values {'x': 'bad bad bad'} are outside of space" in str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_insert_params_bad_results(self, factory):
        """Test that results with from format cannot be saved (trial is registered anyhow)"""
        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100), [dict(name="objective", type="bad bad bad", value=0)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestReserve:
    """Tests for ExperimentClient.reserve"""

    def test_reserve(self, factory):
        """Test reservation of registered trials"""
        if is_rest(factory):
            pytest.skip("API REST does not reserve individual trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            assert trial.status == "reserved"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_dont_exist(self, factory):
        """Verify that unregistered trials cannot be reserved."""
        if is_rest(factory):
            pytest.skip("API REST does not reserve individual trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.reserve(trial)

            assert f"Trial {trial.id} does not exist in database." == str(exc.value)
            assert client._pacemakers == {}

    def test_reserve_reserved_locally(self, caplog, factory):
        """Verify that a trial cannot be reserved twice locally (warning, no exception)"""
        if is_rest(factory):
            pytest.skip("API REST does not reserve individual trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            with caplog.at_level(logging.WARNING):
                client.reserve(trial)

            assert (
                f"Trial {trial.id} is already reserved." == caplog.records[-1].message
            )

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_reserved_remotely(self, factory):
        """Verify that a trial cannot be reserved if already reserved by another process"""
        if is_rest(factory):
            pytest.skip("API REST does not reserve individual trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[2])
            assert trial.status == "interrupted"
            client.reserve(trial)
            remote_pacemaker = client._pacemakers.pop(trial.id)
            assert experiment.get_trial(trial).status == "reserved"

            trial = Trial(**cfg.trials[2])
            assert trial.status == "interrupted"
            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert f"Could not reserve trial {trial.id}." == str(exc.value)

            assert trial.status == "interrupted"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers == {}
            remote_pacemaker.stop()

    def test_reserve_race_condition(self, factory):
        """Verify that race conditions during `reserve` is detected and raises a comprehensible
        error
        """
        if is_rest(factory):
            pytest.skip("API REST does not reserve individual trials")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[0]["id"])
            experiment.set_trial_status(trial, "reserved")
            trial.status = "new"  # Let's pretend it is still available

            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert f"Could not reserve trial {trial.id}." == str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestRelease:
    """Tests for ExperimentClient.release"""

    def test_release(self, factory):
        """Test releasing (to interrupted)"""
        with factory(config, base_trial) as (cfg, experiment, client):
            if is_rest(factory):
                trial = client.suggest()
            else:
                trial = experiment.get_trial(uid=cfg.trials[1]["id"])
                client.reserve(trial)

            pacemaker = client._pacemakers[trial.id]
            client.release(trial)

            if not is_rest(factory):
                assert trial.status == "interrupted"

            assert experiment.get_trial(uid=trial.id).status == "interrupted"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_status(self, factory):
        """Test releasing with a specific status"""
        with factory(config, base_trial) as (cfg, experiment, client):
            if is_rest(factory):
                trial = client.suggest()
            else:
                trial = experiment.get_trial(uid=cfg.trials[1]["id"])
                client.reserve(trial)

            pacemaker = client._pacemakers[trial.id]
            client.release(trial, "broken")

            if not is_rest(factory):
                assert trial.status == "broken"

            assert experiment.get_trial(uid=trial.id).status == "broken"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_invalid_status(self, factory):
        """Test releasing with a specific status"""
        with factory(config, base_trial) as (cfg, experiment, client):
            if is_rest(factory):
                trial = client.suggest()
            else:
                trial = experiment.get_trial(uid=cfg.trials[1]["id"])
                client.reserve(trial)

            with pytest.raises(ValueError) as exc:
                client.release(trial, "mouf mouf")

            assert exc.match("Given status `mouf mouf` not one of")

    def test_release_dont_exist(self, monkeypatch, factory):
        """Verify that unregistered trials cannot be released"""
        with factory(config, base_trial) as (cfg, experiment, client):
            params = copy.deepcopy(cfg.trials[1]["params"])
            params[0]["value"] += 12.12

            trial = Trial(experiment="idontexist", params=params)

            def do_nada(trial, **kwargs):
                """Don't do anything"""
                return None

            monkeypatch.setattr(client, "_release_reservation", do_nada)

            with pytest.raises(ValueError) as exc:
                client.release(trial)

            assert f"Trial {trial.id} does not exist in database." == str(exc.value)
            assert client._pacemakers == {}

    def test_release_race_condition(self, factory):
        """Verify that race conditions during `release` is detected and raises a comprehensible
        error
        """
        if is_rest(factory):
            # REST API does not have the Trial object with its previous status
            pytest.skip("REST API release always works")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            if not is_rest(factory):
                trial = client.get_trial(uid=cfg.trials[1]["id"])
                client.reserve(trial)
            else:
                # trial is now reserved
                trial = client.suggest()

            pacemaker = client._pacemakers[trial.id]
            # Woops! Trial got failed over from another process.
            server_trial = experiment.get_trial(uid=trial.id)
            experiment.set_trial_status(server_trial, "interrupted")
            server_trial.status = "reserved"  # Let's pretend we don't know.

            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert "Reservation for trial {} has been lost before release.".format(
                trial.id
            ) in str(exc.value)
            assert client._pacemakers == {}
            assert not pacemaker.is_alive()

    def test_release_unreserved(self, factory):
        """Verify that unreserved trials cannot be released"""
        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["id"])
            with pytest.raises(AlreadyReleased) as exc:
                client.release(trial)

            assert f"Trial {trial.id} was already released locally." == str(exc.value)

            assert client._pacemakers == {}

    def test_release_already_released_but_incorrectly(self, factory):
        """Verify that incorrectly released trials have its pacemaker stopped properly"""
        with factory(config, base_trial) as (cfg, experiment, client):
            if not is_rest(factory):
                trial = client.get_trial(uid=cfg.trials[1]["id"])
                client.reserve(trial)
            else:
                trial = client.suggest()

            pacemaker = client._pacemakers[trial.id]

            if is_rest(factory):
                ref_trial = experiment.get_trial(uid=trial.id)
                assert ref_trial.status == "reserved"
                experiment.set_trial_status(ref_trial, "interrupted")
                assert ref_trial.status == "interrupted"

            else:
                assert trial.status == "reserved"
                experiment.set_trial_status(trial, "interrupted")
                assert trial.status == "interrupted"

            with pytest.raises(AlreadyReleased) as exc:
                client.release(trial)

            assert f"Trial {trial.id} was already released locally." == str(exc.value)

            assert client._pacemakers == {}
            assert not pacemaker.is_alive()


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestClose:
    """Test close method of the client"""

    def test_close_empty(self, factory):
        """Test client can close when no trial is reserved"""
        with factory(config, base_trial) as (cfg, experiment, client):
            client.close()

    def test_close_with_reserved(self, factory):
        """Test client cannot be closed if trials are reserved."""
        with factory(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                with pytest.raises(RuntimeError) as exc:
                    client.close()

                assert "There is still reserved trials" in str(exc.value)


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestBroken:
    """Test handling of broken trials"""

    def test_broken_trial(self, factory):
        """Test that broken trials are detected"""
        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(RuntimeError):
                with client.suggest() as trial:
                    trial = experiment.get_trial(uid=trial.id)
                    assert trial.status == "reserved"
                    raise RuntimeError("Dummy failure!")

            assert client._pacemakers == {}
            trial = experiment.get_trial(uid=trial.id)
            assert trial.status == "broken"

    def test_interrupted_trial(self, factory):
        """Test that interrupted trials are not set to broken"""
        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:
                    trial = experiment.get_trial(uid=trial.id)
                    assert trial.status == "reserved"
                    raise KeyboardInterrupt

            assert client._pacemakers == {}
            trial = experiment.get_trial(uid=trial.id)
            assert trial.status == "interrupted"

    def test_completed_then_interrupted_trial(self, factory):
        """Test that interrupted trials are not set to broken"""

        with factory(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:

                    ref_trial = experiment.get_trial(uid=trial.id)
                    assert ref_trial.status == "reserved"
                    assert ref_trial.results == []
                    assert ref_trial.objective is None

                    client.observe(
                        trial, [dict(name="objective", type="objective", value=101)]
                    )

                    ref_trial = experiment.get_trial(uid=trial.id)
                    assert ref_trial.objective.value == 101
                    assert ref_trial.status == "completed"

                    raise KeyboardInterrupt

            assert client._pacemakers == {}

            trial = experiment.get_trial(uid=trial.id)
            assert trial.status == "completed"


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestSuggest:
    """Tests for ExperimentClient.suggest"""

    def test_suggest(self, monkeypatch, factory):
        """Verify that suggest reserved available trials."""
        mock_space_iterate(monkeypatch)
        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == 1

            assert len(experiment.fetch_trials()) == 5
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_suggest_new(self, factory):
        """Verify that suggest can create, register and reserved new trials."""
        with factory(config, base_trial) as (cfg, experiment, client):
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

    def test_suggest_race_condition(self, monkeypatch, factory):
        """Verify that race conditions to register new trials is handled"""
        mock_space_iterate(monkeypatch)
        new_value = 50.0

        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            # algo will suggest once an already existing trial
            def amnesia(num=1):
                """Suggest a new value and then always suggest the same"""
                return [format_trials.tuple_to_trial([0], experiment.space)]

            monkeypatch.setattr(experiment.algorithms, "suggest", amnesia)

            assert len(experiment.fetch_trials()) == 1

            with pytest.raises(WaitingForTrials):
                trial = client.suggest()

            assert len(experiment.fetch_trials()) == 1

    def test_suggest_algo_opt_out(self, monkeypatch, factory):
        """Verify that None is returned when algo cannot sample new trials (opting opt)"""

        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        def opt_out(num=1):
            """Never suggest a new trial"""
            return []

        monkeypatch.setattr(orion.core.config.worker, "idle_timeout", -1)

        with factory(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(experiment.algorithms, "suggest", opt_out)

            assert len(experiment.fetch_trials()) == 1

            with pytest.raises(WaitingForTrials):
                client.suggest()

    def test_suggest_is_done(self, factory):
        """Verify that completed experiments cannot suggest new trials"""
        with factory(config, base_trial, statuses=["completed"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            with pytest.raises(CompletedExperiment):
                client.suggest()

    def test_suggest_is_done_context_manager(self, factory):
        """Verify that context manager handles None"""
        with factory(config, base_trial, statuses=["completed"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            with pytest.raises(CompletedExperiment):
                client.suggest()

    def test_suggest_is_broken(self, factory):
        """Verify that broken experiments cannot suggest new trials"""
        with factory(config, base_trial, statuses=["broken"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

    def test_suggest_is_done_race_condition(self, monkeypatch, factory):
        """Verify that inability to suggest because is_done becomes True during produce() is
        handled.
        """
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial, statuses=["completed"] * 5) as (
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

    def test_suggest_reserve_race_condition(self, monkeypatch, factory):
        """Verify that when trials are produced and reserved by a different worker an
        exception is raised

        """
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial, statuses=["completed"] * 5) as (
            cfg,
            experiment,
            client,
        ):

            def produce(pool_size):
                """Set is_done while algo is trying to suggest"""
                return 10

            monkeypatch.setattr(client._producer, "produce", produce)

            assert len(experiment.fetch_trials()) == 5
            assert not client.is_done

            with pytest.raises(ReservationRaceCondition):
                client.suggest()

            assert len(experiment.fetch_trials()) == 5

    def test_suggest_is_broken_race_condition(self, monkeypatch, factory):
        """Verify that experiments that gets broken during local algo.suggest gets properly
        handled
        """
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial, statuses=["broken"] * 1) as (
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

            start_time = time.time()

            assert len(experiment.fetch_trials()) == 1
            assert not client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

            assert time.time() - start_time < 3
            assert len(experiment.fetch_trials()) == 1
            assert client.is_broken

    def test_suggest_hierarchical_space(self, factory):
        """Verify that suggest returns trial with proper hierarchical parameter."""
        exp_config = copy.deepcopy(config)
        exp_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)", "z": "choices(['voici', 'voila', 2])"},
        }
        with factory(exp_config=exp_config, trial_config=base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert len(trial.params) == 2
            assert "x" in trial.params["a"]
            assert "y" in trial.params["b"]
            assert "z" in trial.params["b"]

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()


@pytest.mark.parametrize("factory", factories)
@pytest.mark.usefixtures("version_XYZ")
class TestObserve:
    """Tests for ExperimentClient.observe"""

    def test_observe(self, factory):
        """Verify that `observe()` will update the storage"""
        with factory(config, base_trial) as (cfg, experiment, client):

            if not is_rest(factory):
                trial = Trial(**cfg.trials[1])
                assert trial.results == []
                client.reserve(trial)
            else:
                trial = client.suggest()

            assert experiment.get_trial(uid=trial.id).objective is None
            client.observe(trial, [dict(name="objective", type="objective", value=101)])
            assert experiment.get_trial(uid=trial.id).objective.value == 101

    def test_observe_unreserved(self, factory):
        """Verify that `observe()` will fail on non-reserved trials"""
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])

            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Trial {} had no pacemakers. Was it reserved properly?".format(
                trial.id
            ) == str(exc.value)

    def test_observe_dont_exist(self, factory):
        """Verify that `observe()` will fail on non-registered trials"""
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert f"Trial {trial.id} does not exist in database." == str(exc.value)
            assert client._pacemakers == {}

    def test_observe_bad_results(self, factory):
        """Verify that bad results type is detected and ValueError is raised"""

        with factory(config, base_trial) as (cfg, experiment, client):

            if is_rest(factory):
                trial = client.suggest()
            else:
                trial = Trial(**cfg.trials[1])
                client.reserve(trial)

            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="bad bad bad", value=101)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_observe_race_condition(self, factory):
        """Verify that race condition during `observe()` is detected and raised"""
        if is_rest(factory):
            pytest.skip("This test cannot work on the REST API")
            return

        with factory(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["id"])
            client.reserve(trial)
            experiment.set_trial_status(trial, "interrupted")
            trial.status = "reserved"  # Let's pretend it is still reserved

            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert f"Reservation for trial {trial.id} has been lost." == str(exc.value)
            assert client._pacemakers == {}

    def test_observe_under_with(self, factory):
        with factory(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:

                ref_trial = experiment.get_trial(uid=trial.id)

                assert ref_trial.status == "reserved"
                assert ref_trial.results == []
                assert ref_trial.objective is None

                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

                ref_trial = experiment.get_trial(uid=trial.id)
                assert ref_trial.objective.value == 101
                assert ref_trial.status == "completed"

            if not is_rest(factory):
                assert trial.status == "completed"  # Still completed after __exit__

    def test_observe_with_float(self, factory):
        with factory(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                client.observe(trial, 10.0)

                ref_trial = experiment.get_trial(uid=trial.id)
                assert ref_trial.status == "completed"
                assert ref_trial.objective.name == "objective"
                assert ref_trial.objective.type == "objective"
                assert not client._pacemakers

    def test_observe_with_float_and_name(self, factory):
        with factory(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                client.observe(trial, 10.0, name="custom_objective")

                ref_trial = experiment.get_trial(uid=trial.id)
                assert ref_trial.status == "completed"
                assert ref_trial.objective.name == "custom_objective"
                assert ref_trial.objective.type == "objective"
                assert not client._pacemakers

    def test_observe_with_invalid_type(self, factory):
        with factory(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:

                with pytest.raises(TypeError):
                    client.observe(trial, "invalid")

                ref_trial = experiment.get_trial(uid=trial.id)
                assert ref_trial.status == "reserved"
                assert ref_trial.objective is None


@pytest.mark.parametrize("factory", factories)
def test_executor_receives_correct_worker_count(factory):
    """Check that the client forwards the current number count to the executor"""

    with factory(config, base_trial) as (cfg, experiment, client):
        assert client.executor.n_workers == orion.core.config.worker.n_workers

    with factory(config, base_trial) as (cfg, experiment, client):
        with client.tmp_executor("joblib", n_workers=3, backend="threading"):
            assert client.executor.n_workers == 3


def function(a, b):
    return a + b


@pytest.mark.parametrize("factory", factories)
def test_executor_gets_created_if_not_provided(factory):
    """Check that executors created by the client are cleanup"""
    global config
    conf = copy.deepcopy(config)

    # make sure the executor is not set
    conf.pop("executor", None)
    executor = None

    with factory(config, base_trial) as (cfg, experiment, client):
        executor = client.executor
        assert executor is not None, "Client created an executor"
        assert client._executor_owner is True, "Client own the executor"

    assert client._executor is None, "Client freed the executor"
    assert client._executor_owner is False, "Client does not own the executor"

    # executor was closed and cannot be used
    with pytest.raises(ExecutorClosed):
        executor.submit(function, 2, 2)


@pytest.mark.parametrize("factory", factories)
def test_user_executor_is_not_deleted(factory):
    """Check that executors passed to the client are not cleanup"""

    global config
    conf = copy.deepcopy(config)

    executor = executor_factory.create("joblib", 1)
    conf["executor"] = executor

    with factory(config, base_trial) as (cfg, experiment, client):
        assert client.executor is not None, "Client has an executor"
        assert client._executor_owner is True, "Client does not own the executor"

    future = executor.submit(function, 2, 2)
    assert future.get() == 4, "Executor was not closed & can still be used"


def main(*args, **kwargs):
    return [dict(name="objective", type="objective", value=101)]


def test_run_experiment_twice():
    """"""

    with create_experiment(config, base_trial) as (cfg, experiment, client):
        client.workon(main, max_trials=10)

        client._experiment.max_trials = 20
        client._experiment.algorithms.algorithm.max_trials = 20

        client.workon(main, max_trials=20)
