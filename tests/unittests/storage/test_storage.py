#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import copy
import datetime
import logging
import os
import pickle
import time

import pytest

import orion.core
from orion.core.io.database import DuplicateKeyError
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils.singleton import (
    SingletonAlreadyInstantiatedError,
    SingletonNotInstantiatedError,
    update_singletons,
)
from orion.core.worker.trial import Trial
from orion.storage.base import (
    FailedUpdate,
    MissingArguments,
    Storage,
    get_storage,
    setup_storage,
)
from orion.storage.legacy import Legacy
from orion.storage.track import HAS_TRACK, REASON
from orion.testing import OrionState

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

storage_backends = [None]  # defaults to legacy with PickleDB

if not HAS_TRACK:
    log.warning("Track is not tested because: %s!", REASON)
else:
    storage_backends.append({"type": "track", "uri": "file://${file}?objective=loss"})

base_experiment = {
    "name": "default_name",
    "version": 0,
    "metadata": {
        "user": "default_user",
        "user_script": "abc",
        "priors": {"x": "uniform(0, 10)"},
        "datetime": "2017-11-23T02:00:00",
        "orion_version": "XYZ",
    },
}


base_trial = {
    "experiment": "default_name",
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "submit_time": "2017-11-23T02:00:00",
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [
        {"name": "loss", "type": "objective", "value": 2}  # objective, constraint
    ],
    "params": [
        {"name": "/encoding_layer", "type": "categorical", "value": "rnn"},
        {
            "name": "/decoding_layer",
            "type": "categorical",
            "value": "lstm_with_attention",
        },
    ],
}


def _generate(obj, *args, value):
    if obj is None:
        return None

    obj = copy.deepcopy(obj)
    data = obj

    for arg in args[:-1]:
        data = data[arg]

    data[args[-1]] = value
    return obj


def make_lost_trial(delay=2):
    """Make a lost trial"""
    obj = copy.deepcopy(base_trial)
    obj["status"] = "reserved"
    obj["heartbeat"] = datetime.datetime.utcnow() - datetime.timedelta(
        seconds=orion.core.config.worker.heartbeat * delay
    )
    obj["params"].append(
        {"name": "/index", "type": "categorical", "value": f"lost_trial_{delay}"}
    )
    return obj


all_status = ["completed", "broken", "reserved", "interrupted", "suspended", "new"]


def generate_trials(status=None, heartbeat=None):
    """Generate Trials with different configurations"""
    if status is None:
        status = all_status

    new_trials = [_generate(base_trial, "status", value=s) for s in status]

    if heartbeat:
        for trial in new_trials:
            trial["heartbeat"] = heartbeat

    # make each trial unique
    for i, trial in enumerate(new_trials):
        trial["params"].append({"name": "/index", "type": "categorical", "value": i})

    return new_trials


def generate_experiments():
    """Generate a set of experiments"""
    users = ["a", "b", "c"]
    exps = [_generate(base_experiment, "metadata", "user", value=u) for u in users]
    return [_generate(exp, "name", value=str(i)) for i, exp in enumerate(exps)]


@pytest.mark.usefixtures("setup_pickleddb_database")
def test_setup_storage_default():
    """Test that storage is setup using default config"""
    update_singletons()
    setup_storage()
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)


def test_setup_storage_bad():
    """Test how setup fails when configuring with non-existant backends"""
    update_singletons()
    with pytest.raises(NotImplementedError) as exc:
        setup_storage({"type": "idontexist"})

    assert exc.match("idontexist")


def test_setup_storage_custom():
    """Test setup with local configuration"""
    update_singletons()
    setup_storage(
        {"type": "legacy", "database": {"type": "pickleddb", "host": "test.pkl"}}
    )
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    assert storage._db.host == os.path.abspath("test.pkl")


def test_setup_storage_custom_type_missing():
    """Test setup with local configuration with type missing"""
    update_singletons()
    setup_storage({"database": {"type": "pickleddb", "host": "test.pkl"}})
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    assert storage._db.host == os.path.abspath("test.pkl")


@pytest.mark.usefixtures("setup_pickleddb_database")
def test_setup_storage_custom_legacy_emtpy():
    """Test setup with local configuration with legacy but no config"""
    update_singletons()
    setup_storage({"type": "legacy"})
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    assert storage._db.host == orion.core.config.storage.database.host


def test_setup_storage_bad_override():
    """Test setup with different type than existing singleton"""
    update_singletons()
    setup_storage(
        {"type": "legacy", "database": {"type": "pickleddb", "host": "test.pkl"}}
    )
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    with pytest.raises(SingletonAlreadyInstantiatedError) as exc:
        setup_storage({"type": "track"})

    assert exc.match("A singleton instance of \(type: Storage\)")


@pytest.mark.xfail(reason="Fix this when introducing #135 in v0.2.0")
def test_setup_storage_bad_config_override():
    """Test setup with different config than existing singleton"""
    update_singletons()
    setup_storage({"database": {"type": "pickleddb", "host": "test.pkl"}})
    storage = Storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    with pytest.raises(SingletonAlreadyInstantiatedError):
        setup_storage({"database": {"type": "mongodb"}})


def test_setup_storage_stateless():
    """Test that passed configuration dictionary is not modified by the fonction"""
    update_singletons()
    config = {"database": {"type": "pickleddb", "host": "test.pkl"}}
    passed_config = copy.deepcopy(config)
    setup_storage(passed_config)
    assert config == passed_config


def test_get_storage_uninitiated():
    """Test that get storage fails if no storage singleton exist"""
    update_singletons()
    with pytest.raises(SingletonNotInstantiatedError) as exc:
        get_storage()

    assert exc.match("No singleton instance of \(type: Storage\) was created")


def test_get_storage():
    """Test that get storage gets the singleton"""
    update_singletons()
    setup_storage({"database": {"type": "pickleddb", "host": "test.pkl"}})
    storage = get_storage()
    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, PickledDB)
    assert get_storage() == storage


@pytest.mark.usefixtures("version_XYZ")
@pytest.mark.parametrize("storage", storage_backends)
class TestStorage:
    """Test all storage backend"""

    def test_create_experiment(self, storage):
        """Test create experiment"""
        with OrionState(experiments=[], storage=storage) as cfg:
            storage = cfg.storage()

            storage.create_experiment(base_experiment)

            experiments = storage.fetch_experiments({})
            assert len(experiments) == 1, "Only one experiment in the database"

            experiment = experiments[0]
            assert base_experiment == experiment, "Local experiment and DB should match"

            # Insert it again
            with pytest.raises(DuplicateKeyError):
                storage.create_experiment(base_experiment)

    def test_fetch_experiments(self, storage, name="0", user="a"):
        """Test fetch experiments"""
        with OrionState(experiments=generate_experiments(), storage=storage) as cfg:
            storage = cfg.storage()

            experiments = storage.fetch_experiments({})
            assert len(experiments) == len(cfg.experiments)

            experiments = storage.fetch_experiments(
                {"name": name, "metadata.user": user}
            )
            assert len(experiments) == 1

            experiment = experiments[0]
            assert experiment["name"] == name, "name should match query"
            assert (
                experiment["metadata"]["user"] == user
            ), "user name should match query"

            experiments = storage.fetch_experiments(
                {"name": "-1", "metadata.user": user}
            )
            assert len(experiments) == 0

    def test_update_experiment(self, monkeypatch, storage, name="0", user="a"):
        """Test fetch experiments"""
        with OrionState(experiments=generate_experiments(), storage=storage) as cfg:
            storage = cfg.storage()

            class _Dummy:
                pass

            experiment = cfg.experiments[0]
            mocked_experiment = _Dummy()
            mocked_experiment.id = experiment["_id"]

            storage.update_experiment(mocked_experiment, test=True)
            experiments = storage.fetch_experiments({"_id": experiment["_id"]})
            assert len(experiments) == 1

            fetched_experiment = experiments[0]
            assert fetched_experiment["test"]

            assert (
                "test"
                not in storage.fetch_experiments({"_id": cfg.experiments[1]["_id"]})[0]
            )

            storage.update_experiment(uid=experiment["_id"], test2=True)
            assert storage.fetch_experiments({"_id": experiment["_id"]})[0]["test2"]
            assert (
                "test2"
                not in storage.fetch_experiments({"_id": cfg.experiments[1]["_id"]})[0]
            )

            with pytest.raises(MissingArguments):
                storage.update_experiment()

            with pytest.raises(AssertionError):
                storage.update_experiment(experiment=mocked_experiment, uid="123")

    def test_delete_experiment(self, storage):
        """Test delete one experiment"""
        if storage and storage["type"] == "track":
            pytest.xfail("Track does not support deletion yet.")

        with OrionState(experiments=generate_experiments(), storage=storage) as cfg:
            storage = cfg.storage()

            n_experiments = len(storage.fetch_experiments({}))
            storage.delete_experiment(uid=cfg.experiments[0]["_id"])
            experiments = storage.fetch_experiments({})
            assert len(experiments) == n_experiments - 1
            assert cfg.experiments[0]["_id"] not in [exp["_id"] for exp in experiments]

    def test_register_trial(self, storage):
        """Test register trial"""
        with OrionState(experiments=[base_experiment], storage=storage) as cfg:
            storage = cfg.storage()
            trial1 = storage.register_trial(Trial(**base_trial))
            trial2 = storage.get_trial(trial1)

            assert (
                trial1.to_dict() == trial2.to_dict()
            ), "Trials should match after insert"

    def test_register_duplicate_trial(self, storage):
        """Test register trial"""
        with OrionState(
            experiments=[base_experiment], trials=[base_trial], storage=storage
        ) as cfg:
            storage = cfg.storage()

            with pytest.raises(DuplicateKeyError):
                storage.register_trial(Trial(**base_trial))

    def test_register_lie(self, storage):
        """Test register lie"""
        with OrionState(experiments=[base_experiment], storage=storage) as cfg:
            storage = cfg.storage()
            storage.register_lie(Trial(**base_trial))

    def test_register_lie_fail(self, storage):
        """Test register lie"""
        with OrionState(
            experiments=[base_experiment], lies=[base_trial], storage=storage
        ) as cfg:
            storage = cfg.storage()

            with pytest.raises(DuplicateKeyError):
                storage.register_lie(Trial(**cfg.lies[0]))

    def test_update_trials(self, storage):
        """Test update many trials"""
        with OrionState(
            experiments=[base_experiment],
            trials=generate_trials(status=["completed", "reserved", "reserved"]),
            storage=storage,
        ) as cfg:
            storage = cfg.storage()

            class _Dummy:
                pass

            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.fetch_trials(experiment)
            assert sum(trial.status == "reserved" for trial in trials) == 2
            count = storage.update_trials(
                experiment, where={"status": "reserved"}, status="interrupted"
            )
            assert count == 2
            trials = storage.fetch_trials(experiment)
            assert sum(trial.status == "interrupted" for trial in trials) == 2

    def test_update_trial(self, storage):
        """Test update one trial"""
        with OrionState(
            experiments=[base_experiment], trials=[base_trial], storage=storage
        ) as cfg:
            storage = cfg.storage()

            trial = Trial(**cfg.trials[0])

            assert trial.status != "interrupted"
            storage.update_trial(trial, status="interrupted")
            assert storage.get_trial(trial).status == "interrupted"

    def test_reserve_trial_success(self, storage):
        """Test reserve trial"""
        with OrionState(
            experiments=[base_experiment], trials=[base_trial], storage=storage
        ) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name", version=None)

            trial = storage.reserve_trial(experiment)

            assert trial is not None
            assert trial.status == "reserved"

    def test_reserve_trial_fail(self, storage):
        """Test reserve trial"""
        with OrionState(
            experiments=[base_experiment],
            trials=generate_trials(status=["completed", "reserved"]),
            storage=storage,
        ) as cfg:

            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name", version=None)

            trial = storage.reserve_trial(experiment)
            assert trial is None

    def test_fetch_trials(self, storage):
        """Test fetch experiment trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name", version=None)

            trials1 = storage.fetch_trials(experiment=experiment)
            trials2 = storage.fetch_trials(uid=experiment.id)

            with pytest.raises(MissingArguments):
                storage.fetch_trials()

            with pytest.raises(AssertionError):
                storage.fetch_trials(experiment=experiment, uid="123")

            assert len(trials1) == len(cfg.trials), "trial count should match"
            assert len(trials2) == len(cfg.trials), "trial count should match"

    def test_fetch_trials_with_query(self, storage):
        """Test fetch experiment trials with queries"""
        with OrionState(
            experiments=[base_experiment],
            trials=generate_trials(status=["completed", "reserved", "reserved"]),
            storage=storage,
        ) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name", version=None)

            trials_all = storage.fetch_trials(experiment=experiment)
            trials_completed = storage.fetch_trials(
                experiment=experiment, where={"status": "completed"}
            )
            trials_reserved = storage.fetch_trials(
                experiment=experiment, where={"status": "reserved"}
            )

            assert len(trials_all) == len(cfg.trials), "trial count should match"
            assert len(trials_completed) == 1, "trial count should match"
            assert len(trials_reserved) == 2, "trial count should match"

    def test_delete_all_trials(self, storage):
        """Test delete all trials of an experiment"""
        if storage and storage["type"] == "track":
            pytest.xfail("Track does not support deletion yet.")

        trials = generate_trials()
        trial_from_other_exp = copy.deepcopy(trials[0])
        trial_from_other_exp["experiment"] = "other"
        trials.append(trial_from_other_exp)
        with OrionState(
            experiments=[base_experiment], trials=trials, storage=storage
        ) as cfg:
            storage = cfg.storage()

            # Make sure we have sufficient trials to test deletion
            trials = storage.fetch_trials(uid="default_name")
            assert len(trials) > 2

            count = storage.delete_trials(uid="default_name")
            assert count == len(trials)
            assert storage.fetch_trials(uid="default_name") == []

            # Make sure trials from other experiments were not deleted
            assert len(storage.fetch_trials(uid="other")) == 1

    def test_delete_trials_with_query(self, storage):
        """Test delete experiment trials matching a query"""
        if storage and storage["type"] == "track":
            pytest.xfail("Track does not support deletion yet.")

        trials = generate_trials()
        trial_from_other_exp = copy.deepcopy(trials[0])
        trial_from_other_exp["experiment"] = "other"
        trials.append(trial_from_other_exp)
        with OrionState(
            experiments=[base_experiment], trials=trials, storage=storage
        ) as cfg:
            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name")

            # Make sure we have sufficient trials to test deletion
            status = trials[0]["status"]
            trials = storage.fetch_trials(experiment)
            trials_with_status = storage.fetch_trials_by_status(experiment, status)
            assert len(trials_with_status) > 0
            assert len(trials) > len(trials_with_status)

            # Test deletion
            count = storage.delete_trials(uid="default_name", where={"status": status})
            assert count == len(trials_with_status)
            assert storage.fetch_trials_by_status(experiment, status) == []
            assert len(storage.fetch_trials(experiment)) == len(trials) - len(
                trials_with_status
            )

            # Make sure trials from other experiments were not deleted
            assert len(storage.fetch_trials(uid="other")) == 1

    def test_get_trial(self, storage):
        """Test get trial"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage = cfg.storage()

            trial_dict = cfg.trials[0]

            trial1 = storage.get_trial(trial=Trial(**trial_dict))
            trial2 = storage.get_trial(uid=trial1.id)

            with pytest.raises(MissingArguments):
                storage.get_trial()

            with pytest.raises(AssertionError):
                storage.get_trial(trial=trial1, uid="123")

            assert trial1.to_dict() == trial_dict
            assert trial2.to_dict() == trial_dict

    def test_fetch_lost_trials(self, storage):
        """Test update heartbeat"""
        lost_trials = [
            make_lost_trial(2),  # Is not lost for long enough to be catched
            make_lost_trial(10),
        ]  # Is lost for long enough to be catched
        # Force recent heartbeat to avoid mixing up with lost trials.
        trials = lost_trials + generate_trials(heartbeat=datetime.datetime.utcnow())
        with OrionState(
            experiments=[base_experiment], trials=trials, storage=storage
        ) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.fetch_lost_trials(experiment)

            assert len(trials) == 1

    def test_change_status_success(self, storage):
        """Change the status of a Trial"""

        def check_status_change(new_status):
            with OrionState(
                experiments=[base_experiment], trials=generate_trials(), storage=storage
            ) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, "was not able to retrieve trial for test"

                get_storage().set_trial_status(trial, status=new_status)
                assert (
                    trial.status == new_status
                ), "Trial status should have been updated locally"

                trial = get_storage().get_trial(trial)
                assert (
                    trial.status == new_status
                ), "Trial status should have been updated in the storage"

        check_status_change("completed")
        check_status_change("broken")
        check_status_change("reserved")
        check_status_change("interrupted")
        check_status_change("suspended")
        check_status_change("new")

    def test_change_status_invalid(self, storage):
        """Attempt to change the status of a Trial with an invalid one"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            trial = get_storage().get_trial(cfg.get_trial(0))
            assert trial is not None, "Was not able to retrieve trial for test"

            with pytest.raises(ValueError) as exc:
                get_storage().set_trial_status(trial, status="moo")

            assert exc.match("Given status `moo` not one of")

    def test_change_status_failed_update(self, storage):
        """Change the status of a Trial"""

        def check_status_change(new_status):
            with OrionState(
                experiments=[base_experiment], trials=generate_trials(), storage=storage
            ) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, "Was not able to retrieve trial for test"
                assert trial.status != new_status

                if trial.status == new_status:
                    return

                with pytest.raises(FailedUpdate):
                    trial.status = new_status
                    get_storage().set_trial_status(trial, status=new_status)

        check_status_change("completed")
        check_status_change("broken")
        check_status_change("reserved")
        check_status_change("interrupted")
        check_status_change("suspended")

    def test_change_status_success_thanks_to_was(self, storage):
        """Change the status of a Trial requesting the correct previous state, although local
        one is not up-to-date.
        """

        def check_status_change(new_status):
            with OrionState(
                experiments=[base_experiment], trials=generate_trials(), storage=storage
            ) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, "Was not able to retrieve trial for test"
                assert trial.status != new_status

                if trial.status == new_status:
                    return

                correct_status = trial.status
                trial.status = "broken"
                assert correct_status != "broken"
                with pytest.raises(FailedUpdate):
                    get_storage().set_trial_status(trial, status=new_status)

                get_storage().set_trial_status(
                    trial, status=new_status, was=correct_status
                )

        check_status_change("completed")
        check_status_change("broken")
        check_status_change("reserved")
        check_status_change("interrupted")
        check_status_change("suspended")

    def test_change_status_failed_update_because_of_was(self, storage):
        """Change the status of a Trial requesting the wrong previous state."""

        def check_status_change(new_status):
            with OrionState(
                experiments=[base_experiment], trials=generate_trials(), storage=storage
            ) as cfg:
                trial = get_storage().get_trial(cfg.get_trial(0))
                assert trial is not None, "Was not able to retrieve trial for test"
                assert trial.status != new_status

                if trial.status == new_status:
                    return

                with pytest.raises(FailedUpdate):
                    get_storage().set_trial_status(
                        trial, status=new_status, was=new_status
                    )

        check_status_change("completed")
        check_status_change("broken")
        check_status_change("reserved")
        check_status_change("interrupted")
        check_status_change("suspended")

    def test_fetch_pending_trials(self, storage):
        """Test fetch pending trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.fetch_pending_trials(experiment)

            count = 0
            for trial in cfg.trials:
                if trial["status"] in {"new", "suspended", "interrupted"}:
                    count += 1

            assert len(trials) == count
            for trial in trials:
                assert trial.status in {"new", "suspended", "interrupted"}

    def test_fetch_noncompleted_trials(self, storage):
        """Test fetch non completed trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage = cfg.storage()

            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.fetch_noncompleted_trials(experiment)

            count = 0
            for trial in cfg.trials:
                if trial["status"] != "completed":
                    count += 1

            for trial in trials:
                assert trial.status != "completed"

            assert len(trials) == count

    def test_fetch_trials_by_status(self, storage):
        """Test fetch completed trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial["status"] == "completed":
                    count += 1

            storage = cfg.storage()
            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.fetch_trials_by_status(experiment, "completed")

            assert len(trials) == count
            for trial in trials:
                assert trial.status == "completed", trial

    def test_count_completed_trials(self, storage):
        """Test count completed trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial["status"] == "completed":
                    count += 1

            storage = cfg.storage()

            experiment = cfg.get_experiment("default_name", version=None)
            trials = storage.count_completed_trials(experiment)
            assert trials == count

    def test_count_broken_trials(self, storage):
        """Test count broken trials"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            count = 0
            for trial in cfg.trials:
                if trial["status"] == "broken":
                    count += 1

            storage = cfg.storage()

            experiment = cfg.get_experiment("default_name", version=None)

            trials = storage.count_broken_trials(experiment)

            assert trials == count

    def test_update_heartbeat(self, storage):
        """Test update heartbeat"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage_name = storage
            storage = cfg.storage()

            exp = cfg.get_experiment("default_name")
            trial1 = storage.fetch_trials_by_status(exp, status="reserved")[0]
            trial1b = copy.deepcopy(trial1)

            storage.update_heartbeat(trial1)

            trial2 = storage.get_trial(trial1)

            # this check that heartbeat is the correct type and that it was updated prior to now
            assert trial1b.heartbeat is None
            assert trial1.heartbeat is None
            assert trial2.heartbeat is not None

            # Sleep a bit, because fast CPUs make this test fail
            time.sleep(0.1)
            assert trial2.heartbeat < datetime.datetime.utcnow()

            if storage_name is None:
                trial3 = storage.fetch_trials_by_status(exp, status="completed")[0]
                storage.update_heartbeat(trial3)

                assert (
                    trial3.heartbeat is None
                ), "Legacy does not update trials with a status different from reserved"

    def test_serializable(self, storage):
        """Test storage can be serialized"""
        with OrionState(
            experiments=[base_experiment], trials=generate_trials(), storage=storage
        ) as cfg:
            storage = cfg.storage()
            serialized = pickle.dumps(storage)
            deserialized = pickle.loads(serialized)
            assert storage.fetch_experiments({}) == deserialized.fetch_experiments({})
