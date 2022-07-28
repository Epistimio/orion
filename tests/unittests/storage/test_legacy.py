#!/usr/bin/env python
"""Collection of tests for :mod:`orion.storage`."""
import copy
import logging
import os

import pytest

from orion.core.io.database.pickleddb import PickledDB
from orion.core.worker.trial import Trial
from orion.storage.base import FailedUpdate
from orion.storage.legacy import setup_database
from orion.testing import OrionState

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


base_experiment = {
    "name": "default_name",
    "version": 0,
    "metadata": {
        "user": "default_user",
        "user_script": "abc",
        "datetime": "2017-11-23T02:00:00",
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


mongodb_config = {
    "database": {
        "type": "MongoDB",
        "name": "orion_test",
        "username": "user",
        "password": "pass",
    }
}

db_backends = [{"type": "legacy", "database": mongodb_config}]


@pytest.mark.usefixtures("orionstate")
def test_setup_database_default(monkeypatch):
    """Test that database is setup using default config"""

    database = setup_database()
    assert isinstance(database, PickledDB)


def test_setup_database_bad():
    """Test how setup fails when configuring with non-existent backends"""

    with pytest.raises(NotImplementedError) as exc:
        setup_database({"type": "idontexist"})

    assert exc.match("idontexist")


def test_setup_database_custom():
    """Test setup with local configuration"""

    database = setup_database({"type": "pickleddb", "host": "test.pkl"})

    assert isinstance(database, PickledDB)
    assert database.host == os.path.abspath("test.pkl")


def test_setup_database_bad_override():
    """Test setup with different type than existing singleton"""

    database = setup_database({"type": "pickleddb", "host": "test.pkl"})
    assert isinstance(database, PickledDB)


def test_setup_database_bad_config_override():
    """Test setup with different config than existing singleton"""

    database = setup_database({"type": "pickleddb", "host": "test.pkl"})
    assert isinstance(database, PickledDB)


def test_get_database():
    """Test that get database gets the singleton"""

    database = setup_database({"type": "pickleddb", "host": "test.pkl"})
    assert isinstance(database, PickledDB)


class TestLegacyStorage:
    """Test Legacy Storage retrieve result mechanic separately"""

    def test_push_trial_results(self, storage=None):
        """Successfully push a completed trial into database."""
        reserved_trial = copy.deepcopy(base_trial)
        reserved_trial["status"] = "reserved"
        with OrionState(
            experiments=[], trials=[reserved_trial], storage=storage
        ) as cfg:
            storage = cfg.storage
            trial = storage.get_trial(Trial(**reserved_trial))
            results = [Trial.Result(name="loss", type="objective", value=2)]
            trial.results = results
            assert storage.push_trial_results(trial), "should update successfully"

            trial2 = storage.get_trial(trial)
            assert trial2.results == results

    def test_push_trial_results_unreserved(self, storage=None):
        """Successfully push a completed trial into database."""
        with OrionState(experiments=[], trials=[base_trial], storage=storage) as cfg:
            storage = cfg.storage
            trial = storage.get_trial(Trial(**base_trial))
            results = [Trial.Result(name="loss", type="objective", value=2)]
            trial.results = results
            with pytest.raises(FailedUpdate):
                storage.push_trial_results(trial)
