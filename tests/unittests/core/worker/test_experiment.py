#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.experiment`."""

import copy
import datetime
import inspect
import json
import logging
import pickle
import tempfile

import pandas
import pytest

import orion.core
import orion.core.utils.backward as backward
import orion.core.worker.experiment
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.exceptions import UnsupportedOperation
from orion.core.worker.experiment import Experiment
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage
from orion.testing import OrionState


@pytest.fixture()
def new_config(random_dt):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name="supernaekei",
        # refers is missing on purpose
        metadata={
            "user": "tsirif",
            "orion_version": 0.1,
            "user_script": "tests/functional/demo/black_box.py",
            "user_config": "abs_path/hereitis.yaml",
            "user_args": ["--mini-batch~uniform(32, 256, discrete=True)"],
            "VCS": {
                "type": "git",
                "is_dirty": False,
                "HEAD_sha": "fsa7df7a8sdf7a8s7",
                "active_branch": None,
                "diff_sha": "diff",
            },
        },
        version=1,
        pool_size=10,
        max_trials=1000,
        max_broken=5,
        working_dir=None,
        algorithms={"dumbalgo": {}},
        producer={"strategy": "NoParallelStrategy"},
        # attrs starting with '_' also
        # _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored="asdfa",
    )

    backward.populate_space(new_config)

    return new_config


@pytest.fixture
def parent_version_config():
    """Return a configuration for an experiment."""
    config = dict(
        _id="parent_config",
        name="old_experiment",
        version=1,
        algorithms="random",
        metadata={
            "user": "corneauf",
            "datetime": datetime.datetime.utcnow(),
            "user_args": ["--x~normal(0,1)"],
        },
    )

    backward.populate_space(config)

    return config


@pytest.fixture
def child_version_config(parent_version_config):
    """Return a configuration for an experiment."""
    config = copy.deepcopy(parent_version_config)
    config["_id"] = "child_config"
    config["version"] = 2
    config["refers"] = {"parent_id": "parent_config"}
    config["metadata"]["datetime"] = datetime.datetime.utcnow()
    config["metadata"]["user_args"].append("--y~+normal(0,1)")
    backward.populate_space(config)
    return config


def _generate(obj, *args, value):
    if obj is None:
        return None

    obj = copy.deepcopy(obj)
    data = obj

    for arg in args[:-1]:
        data = data[arg]

    data[args[-1]] = value
    return obj


base_trial = {
    "experiment": 0,
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


def generate_trials(status):
    """Generate Trials with different configurations"""
    new_trials = [_generate(base_trial, "status", value=s) for s in status]

    for i, trial in enumerate(new_trials):
        if trial["status"] != "new":
            trial["start_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    for i, trial in enumerate(new_trials):
        if trial["status"] == "completed":
            trial["end_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    # make each trial unique
    for i, trial in enumerate(new_trials):
        trial["results"][0]["value"] = i
        trial["params"].append({"name": "/index", "type": "categorical", "value": i})

    return new_trials


def assert_protocol(exp, storage):
    """Transitional method to move away from mongodb"""
    assert exp._storage._db is storage._db


def count_experiment(exp):
    """Transitional method to move away from mongodb"""
    return exp._storage._db.count("experiments")


@pytest.fixture()
def space():
    """Build a space object"""
    return SpaceBuilder().build({"/index": "uniform(0, 10)"})


@pytest.fixture()
def algorithm(space):
    """Build a dumb algo object"""
    return PrimaryAlgo(space, "dumbalgo")


class TestReserveTrial(object):
    """Calls to interface `Experiment.reserve_trial`."""

    @pytest.mark.usefixtures("setup_pickleddb_database")
    def test_reserve_none(self):
        """Find nothing, return None."""
        with OrionState(experiments=[], trials=[]):
            exp = Experiment("supernaekei", mode="x")
            trial = exp.reserve_trial()
            assert trial is None

    def test_reserve_success(self, random_dt):
        """Successfully find new trials in db and reserve the first one"""
        storage_config = {"type": "legacy", "database": {"type": "EphemeralDB"}}
        with OrionState(
            trials=generate_trials(["new", "reserved"]), storage=storage_config
        ) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]

            trial = exp.reserve_trial()

            # Trials are sorted according to hash and 'new' gets position second
            cfg.trials[1]["status"] = "reserved"
            cfg.trials[1]["start_time"] = random_dt
            cfg.trials[1]["heartbeat"] = random_dt

            assert trial.to_dict() == cfg.trials[1]

    def test_reserve_when_exhausted(self):
        """Return None once all the trials have been allocated"""
        stati = ["new", "reserved", "interrupted", "completed", "broken"]
        with OrionState(trials=generate_trials(stati)) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]
            assert exp.reserve_trial() is not None
            assert exp.reserve_trial() is not None
            assert exp.reserve_trial() is None

    def test_fix_lost_trials(self):
        """Test that a running trial with an old heartbeat is set to interrupted."""
        trial = copy.deepcopy(base_trial)
        trial["status"] = "reserved"
        trial["heartbeat"] = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=60 * 10
        )
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]

            assert len(exp.fetch_trials_by_status("reserved")) == 1
            exp.fix_lost_trials()
            assert len(exp.fetch_trials_by_status("reserved")) == 0

    def test_fix_only_lost_trials(self):
        """Test that an old trial is set to interrupted but not a recent one."""
        lost_trial, running_trial = generate_trials(["reserved"] * 2)
        lost_trial["heartbeat"] = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=60 * 10
        )
        running_trial["heartbeat"] = datetime.datetime.utcnow()

        with OrionState(trials=[lost_trial, running_trial]) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]

            assert len(exp.fetch_trials_by_status("reserved")) == 2

            exp.fix_lost_trials()

            reserved_trials = exp.fetch_trials_by_status("reserved")
            assert len(reserved_trials) == 1
            assert reserved_trials[0].to_dict()["params"] == running_trial["params"]

            failedover_trials = exp.fetch_trials_by_status("interrupted")
            assert len(failedover_trials) == 1
            assert failedover_trials[0].to_dict()["params"] == lost_trial["params"]

    def test_fix_lost_trials_race_condition(self, monkeypatch, caplog):
        """Test that a lost trial fixed by a concurrent process does not cause error."""
        trial = copy.deepcopy(base_trial)
        trial["status"] = "interrupted"
        trial["heartbeat"] = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=60 * 10
        )
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]

            assert len(exp.fetch_trials_by_status("interrupted")) == 1

            assert len(exp._storage.fetch_lost_trials(exp)) == 0

            def fetch_lost_trials(self, query):
                trial_object = Trial(**trial)
                trial_object.status = "reserved"
                return [trial_object]

            # Force the fetch of a trial marked as reserved (and lost) while actually interrupted
            # (as if already failed-over by another process).
            with monkeypatch.context() as m:
                m.setattr(
                    exp._storage.__class__, "fetch_lost_trials", fetch_lost_trials
                )

                assert len(exp._storage.fetch_lost_trials(exp)) == 1

                with caplog.at_level(logging.DEBUG):
                    exp.fix_lost_trials()

            assert caplog.records[-1].levelname == "DEBUG"
            assert caplog.records[-1].msg == "failed"
            assert len(exp.fetch_trials_by_status("interrupted")) == 1
            assert len(exp.fetch_trials_by_status("reserved")) == 0

    def test_fix_lost_trials_configurable_hb(self):
        """Test that heartbeat is correctly being configured."""
        trial = copy.deepcopy(base_trial)
        trial["status"] = "reserved"
        trial["heartbeat"] = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=60 * 2
        )
        with OrionState(trials=[trial]) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]

            assert len(exp.fetch_trials_by_status("reserved")) == 1

            orion.core.config.worker.heartbeat = 60 * 2

            exp.fix_lost_trials()

            assert len(exp.fetch_trials_by_status("reserved")) == 1

            orion.core.config.worker.heartbeat = 60 * 2 / 10.0

            exp.fix_lost_trials()

            assert len(exp.fetch_trials_by_status("reserved")) == 0


def test_update_completed_trial(random_dt):
    """Successfully push a completed trial into database."""
    with OrionState(trials=generate_trials(["new"])) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        trial = exp.reserve_trial()

        results_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="results_", suffix=".log", dir=".", delete=True
        )

        # Generate fake result
        with open(results_file.name, "w") as file:
            json.dump([{"name": "loss", "type": "objective", "value": 2}], file)
        # --

        exp.update_completed_trial(trial, results_file=results_file)

        yo = get_storage().fetch_trials(exp)[0].to_dict()

        assert len(yo["results"]) == len(trial.results)
        assert yo["results"][0] == trial.results[0].to_dict()
        assert yo["status"] == "completed"
        assert yo["end_time"] == random_dt

        results_file.close()


@pytest.mark.usefixtures("with_user_tsirif")
def test_register_trials(random_dt):
    """Register a list of newly proposed trials/parameters."""
    with OrionState():
        exp = Experiment("supernaekei", mode="x")
        exp._id = 0

        trials = [
            Trial(params=[{"name": "a", "type": "integer", "value": 5}]),
            Trial(params=[{"name": "b", "type": "integer", "value": 6}]),
        ]
        for trial in trials:
            exp.register_trial(trial)

        yo = list(map(lambda trial: trial.to_dict(), get_storage().fetch_trials(exp)))
        assert len(yo) == len(trials)
        assert yo[0]["params"] == list(map(lambda x: x.to_dict(), trials[0]._params))
        assert yo[1]["params"] == list(map(lambda x: x.to_dict(), trials[1]._params))
        assert yo[0]["status"] == "new"
        assert yo[1]["status"] == "new"
        assert yo[0]["submit_time"] == random_dt
        assert yo[1]["submit_time"] == random_dt


class TestToPandas:
    """Test suite for ``Experiment.to_pandas``"""

    def test_empty(self, space):
        """Test panda frame creation when there is no trials"""
        with OrionState():
            exp = Experiment("supernaekei", mode="x")
            exp.space = space
            assert exp.to_pandas().shape == (0, 8)
            assert list(exp.to_pandas().columns) == [
                "id",
                "experiment_id",
                "status",
                "suggested",
                "reserved",
                "completed",
                "objective",
                "/index",
            ]

    def test_data(self, space):
        """Verify the data in the panda frame is coherent with database"""
        with OrionState(
            trials=generate_trials(["new", "reserved", "completed"])
        ) as cfg:
            exp = Experiment("supernaekei", mode="x")
            exp._id = cfg.trials[0]["experiment"]
            exp.space = space
            df = exp.to_pandas()
            assert df.shape == (3, 8)
            assert list(df["id"]) == [trial["_id"] for trial in cfg.trials]
            assert all(df["experiment_id"] == exp._id)
            assert list(df["status"]) == ["completed", "reserved", "new"]
            assert list(df["suggested"]) == [
                trial["submit_time"] for trial in cfg.trials
            ]
            assert df["reserved"][0] == cfg.trials[0]["start_time"]
            assert df["reserved"][1] == cfg.trials[1]["start_time"]
            assert df["reserved"][2] is pandas.NaT
            assert df["completed"][0] == cfg.trials[0]["end_time"]
            assert df["completed"][1] is pandas.NaT
            assert df["completed"][2] is pandas.NaT
            assert list(df["objective"]) == [2, 1, 0]
            assert list(df["/index"]) == [2, 1, 0]


def test_fetch_all_trials():
    """Fetch a list of all trials"""
    with OrionState(trials=generate_trials(["new", "reserved", "completed"])) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        trials = list(map(lambda trial: trial.to_dict(), exp.fetch_trials({})))
        assert trials == cfg.trials


def test_fetch_pending_trials():
    """Fetch a list of the trials that are pending

    trials.status in ['new', 'interrupted', 'suspended']
    """
    pending_stati = ["new", "interrupted", "suspended"]
    stati = pending_stati + ["completed", "broken", "reserved"]
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        trials = exp.fetch_pending_trials()
        assert len(trials) == 3
        assert set(trial.status for trial in trials) == set(pending_stati)


def test_fetch_non_completed_trials():
    """Fetch a list of the trials that are not completed

    trials.status in ['new', 'interrupted', 'suspended', 'broken']
    """
    non_completed_stati = ["new", "interrupted", "suspended", "reserved"]
    stati = non_completed_stati + ["completed"]
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        trials = exp.fetch_noncompleted_trials()
        assert len(trials) == 4
        assert set(trial.status for trial in trials) == set(non_completed_stati)


def test_is_done_property_with_pending(algorithm):
    """Check experiment stopping conditions when there is pending trials."""
    completed = ["completed"] * 10
    reserved = ["reserved"] * 5
    with OrionState(trials=generate_trials(completed + reserved)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp.algorithms = algorithm
        exp.max_trials = 10

        assert exp.is_done

        exp.max_trials = 15

        # There is only 10 completed trials
        assert not exp.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done but 5 trials are pending
        assert not exp.is_done


def test_is_done_property_no_pending(algorithm):
    """Check experiment stopping conditions when there is no pending trials."""
    completed = ["completed"] * 10
    broken = ["broken"] * 5
    with OrionState(trials=generate_trials(completed + broken)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp.algorithms = algorithm

        exp.max_trials = 15

        # There is only 10 completed trials and algo not done.
        assert not exp.is_done

        exp.algorithms.algorithm.done = True

        # Algorithm is done and no pending trials
        assert exp.is_done


def test_broken_property():
    """Check experiment stopping conditions for maximum number of broken."""
    MAX_BROKEN = 5

    stati = (["reserved"] * 10) + (["broken"] * (MAX_BROKEN - 1))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp.max_broken = MAX_BROKEN

        assert not exp.is_broken

    stati = (["reserved"] * 10) + (["broken"] * (MAX_BROKEN))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp.max_broken = MAX_BROKEN

        assert exp.is_broken


def test_configurable_broken_property():
    """Check if max_broken changes after configuration."""
    MAX_BROKEN = 5

    stati = (["reserved"] * 10) + (["broken"] * (MAX_BROKEN))
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp.max_broken = MAX_BROKEN

        assert exp.is_broken

        exp.max_broken += 1

        assert not exp.is_broken


def test_experiment_stats():
    """Check that property stats is returning a proper summary of experiment's results."""
    NUM_COMPLETED = 3
    stati = (["completed"] * NUM_COMPLETED) + (["reserved"] * 2)
    with OrionState(trials=generate_trials(stati)) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]
        exp.metadata = {"datetime": datetime.datetime.utcnow()}
        stats = exp.stats
        assert stats["trials_completed"] == NUM_COMPLETED
        assert stats["best_trials_id"] == cfg.trials[3]["_id"]
        assert stats["best_evaluation"] == 0
        assert stats["start_time"] == exp.metadata["datetime"]
        assert stats["finish_time"] == cfg.trials[0]["end_time"]
        assert stats["duration"] == stats["finish_time"] - stats["start_time"]
        assert len(stats) == 6


def test_experiment_pickleable():
    """Test experiment instance is pickleable"""

    with OrionState(trials=generate_trials(["new"])) as cfg:
        exp = Experiment("supernaekei", mode="x")
        exp._id = cfg.trials[0]["experiment"]

        exp_trials = exp.fetch_trials()

        assert len(exp_trials) > 0

        exp_bytes = pickle.dumps(exp)

        new_exp = pickle.loads(exp_bytes)

        assert [trial.to_dict() for trial in exp_trials] == [
            trial.to_dict() for trial in new_exp.fetch_trials()
        ]


read_only_methods = [
    "algorithms",
    "configuration",
    "fetch_lost_trials",
    "fetch_pending_trials",
    "fetch_noncompleted_trials",
    "fetch_trials",
    "fetch_trials_by_status",
    "get_trial",
    "id",
    "is_broken",
    "is_done",
    "max_broken",
    "max_trials",
    "metadata",
    "name",
    "pool_size",
    "producer",
    "refers",
    "retrieve_result",
    "space",
    "stats",
    "to_pandas",
    "version",
    "working_dir",
]
read_write_only_methods = [
    "fix_lost_trials",
    "register_lie",
    "register_trial",
    "set_trial_status",
    "update_completed_trial",
    "duplicate_pending_trials",
]
execute_only_methods = [
    "reserve_trial",
]

ignore = ["non_branching_attrs", "mode", "node"]

dummy_trial = Trial(**_generate(base_trial, "status", value="reserved"))

stati = (["completed"] * 3) + (["reserved"] * 2)
trials = generate_trials(stati)
running_trial = Trial(**trials[-1])

kwargs = {
    "fetch_trials_by_status": {"status": "completed"},
    "get_trial": {"uid": 0},
    "retrieve_result": {"trial": dummy_trial},
    "register_lie": {"lying_trial": dummy_trial},
    "register_trial": {"trial": dummy_trial},
    "set_trial_status": {"trial": dummy_trial, "status": "interrupted"},
    "update_completed_trial": {"trial": running_trial},
}


def test_coverage_of_access_tests():
    """Make sure all attributes and methods are tested to avoid missing access-restricted methods"""
    # TODO, make sure we test all methods.
    all_items = filter(lambda a: not a.startswith("_"), dir(Experiment))
    assert set(all_items) == set(
        read_only_methods + read_write_only_methods + execute_only_methods + ignore
    )


def compare_supported(attr_name, restricted_exp, execution_exp):
    restricted_attr = getattr(restricted_exp, attr_name)
    execution_attr = getattr(execution_exp, attr_name)

    if inspect.ismethod(restricted_attr):
        restricted_attr = restricted_attr(**kwargs.get(attr_name, {}))
        execution_attr = execution_attr(**kwargs.get(attr_name, {}))

    if attr_name == "to_pandas":
        pandas.testing.assert_frame_equal(restricted_attr, execution_attr)
    else:
        assert restricted_attr == execution_attr


def compare_unsupported(attr_name, restricted_exp, execution_exp):
    restricted_attr = getattr(restricted_exp, attr_name)
    execution_attr = getattr(execution_exp, attr_name)

    # So far we only have restricted methods. We will need to modify this test if we ever add
    # restricted properties.
    assert inspect.ismethod(restricted_attr), attr_name

    execution_attr = execution_attr(**kwargs.get(attr_name, {}))
    with pytest.raises(UnsupportedOperation) as exc:
        restricted_attr = restricted_attr(**kwargs.get(attr_name, {}))
    assert exc.match(f"to execute `{attr_name}()")


def create_experiment(mode, space, algorithm):
    experiment = Experiment("supernaekei", mode=mode)
    experiment.space = space
    experiment.algorithms = algorithm
    experiment.max_broken = 5
    experiment.max_trials = 5
    return experiment


class TestReadOnly:
    """Test Experiment access rights in readonly mode"""

    def test_read_only_methods(self, space, algorithm):
        with OrionState(trials=trials) as cfg:
            read_only_exp = create_experiment("r", space, algorithm)
            execution_exp = create_experiment("x", space, algorithm)

            for method in read_only_methods:
                compare_supported(method, read_only_exp, execution_exp)

    def test_read_write_methods(self):
        with OrionState(trials=trials) as cfg:
            read_only_exp = create_experiment("r", space, algorithm)
            execution_exp = create_experiment("x", space, algorithm)
            for method in read_write_only_methods + execute_only_methods:
                compare_unsupported(method, read_only_exp, execution_exp)


class TestReadWriteOnly:
    """Test Experiment access rights in read/write only mode"""

    def test_read_only_methods(self, space, algorithm):
        with OrionState(trials=trials) as cfg:
            read_only_exp = create_experiment("w", space, algorithm)
            execution_exp = create_experiment("x", space, algorithm)

            for method in read_only_methods:
                compare_supported(method, read_only_exp, execution_exp)

    def test_execution_methods(self):
        with OrionState(trials=trials) as cfg:
            read_only_exp = create_experiment("w", space, algorithm)
            execution_exp = create_experiment("x", space, algorithm)
            for method in execute_only_methods:
                compare_unsupported(method, read_only_exp, execution_exp)
