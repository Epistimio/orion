#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import contextlib
import copy
import threading
import time

import pytest

from orion.core.io.experiment_builder import build
from orion.core.worker.producer import Producer
from orion.testing import OrionState, generate_trials


def produce_lies(producer):
    """Wrap production of lies outside of `Producer.update`"""
    return producer._produce_lies(producer.experiment.fetch_noncompleted_trials())


def update_algorithm(producer):
    """Wrap update of algorithm outside of `Producer.update`"""
    return producer._update_algorithm(
        producer.experiment.fetch_trials_by_status("completed")
    )


base_experiment = {
    "name": "default_name",
    "version": 0,
    "space": {"x": "uniform(0, 10, discrete=True)"},
    "metadata": {
        "user": "default_user",
        "user_script": "abc",
        "datetime": "2017-11-23T02:00:00",
        "orion_version": "XYZ",
    },
    "algorithms": {
        "dumbalgo": {
            "value": (5,),
            "scoring": 0,
            "judgement": None,
            "suspend": False,
            "done": False,
            "seed": None,
        }
    },
}

pytestmark = pytest.mark.usefixtures("version_XYZ")


@contextlib.contextmanager
def create_producer():
    """Return a setup `Producer`."""
    # make init done

    with OrionState(
        experiments=[base_experiment],
        trials=generate_trials(exp_config=base_experiment),
    ) as cfg:
        experiment = cfg.get_experiment(name="default_name")

        experiment.algorithms.algorithm.possible_values = [(v,) for v in range(0, 11)]
        experiment.algorithms.seed_rng(0)
        experiment.max_trials = 20
        experiment.algorithms.algorithm.max_trials = 20

        producer = Producer(experiment)
        yield producer, cfg.storage


def test_produce():
    """Test new trials are properly produced"""
    with create_producer() as (producer, _):
        algorithm = producer.experiment.algorithms
        possible_values = [(1,)]
        algorithm.algorithm.possible_values = possible_values

        producer.produce(1)

        # Algorithm was ordered to suggest some trials
        num_new_points = algorithm.algorithm._num
        assert num_new_points == 1  # pool size

        algorithm.algorithm._suggested[0].params["x"] == possible_values[0][0]


def test_register_new_trials():
    """Test new trials are properly registered"""
    with create_producer() as (producer, storage):
        trials_in_db_before = len(storage._fetch_trials({}))
        new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

        algorithm = producer.experiment.algorithms
        possible_values = [(1,)]
        algorithm.algorithm.possible_values = possible_values

        assert producer.produce(1) == 1

        # Algorithm was ordered to suggest some trials
        num_new_points = algorithm.algorithm._num
        assert num_new_points == 1  # pool size

        algorithm.algorithm._suggested[0].params["x"] == possible_values[0][0]

        # `num_new_points` new trials were registered at database
        assert len(storage._fetch_trials({})) == trials_in_db_before + 1
        assert (
            len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 1
        )
        new_trials = list(storage._fetch_trials({"status": "new"}))
        assert new_trials[0].experiment == producer.experiment.id
        assert new_trials[0].start_time is None
        assert new_trials[0].end_time is None
        assert new_trials[0].results == []
        assert new_trials[0].params == {
            "x": 1,
        }


@pytest.mark.skip("How do we test concurrent producers?")
def test_concurent_producers(monkeypatch):
    """Test concurrent production of new trials."""
    with create_producer() as (producer, storage):
        trials_in_db_before = len(storage._fetch_trials({}))
        new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

        producer.experiment.algorithms.algorithm.possible_values = [(1,)]
        # Make sure it starts from index 0
        producer.experiment.algorithms.seed_rng(0)

        second_producer = Producer(producer.experiment)
        second_producer.experiment = copy.deepcopy(producer.experiment)

        sleep = 0.5

        def suggest(self, num):
            time.sleep(sleep)
            return producer.experiment.algorithms.algorithm.possible_values[0]

        monkeypatch.setattr(
            producer.experiment.algorithms.algorithm, "suggest", suggest
        )

        pool = threading.Pool(2)
        first_result = pool.apply_async(producer.produce)
        second_result = pool.apply_async(second_producer.produce, dict(timeout=0))

        assert first_result.get(sleep * 5) == 1

        # TODO: Use Oríon's custom AcquireLockTimeoutError
        with pytest.raises(TimeoutError):
            second_result.get(sleep * 5)

        # `num_new_trials` new trials were registered at database
        assert len(storage._fetch_trials({})) == trials_in_db_before + 1
        assert (
            len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 1
        )
        random_dt = NotImplemented  # todo: undefined variable.
        new_trials = list(
            storage._fetch_trials({"status": "new", "submit_time": random_dt})
        )
        assert new_trials[0].experiment == producer.experiment.id
        assert new_trials[0].start_time is None
        assert new_trials[0].end_time is None
        assert new_trials[0].results == []
        assert new_trials[0].params == {
            "/decoding_layer": "gru",
            "/encoding_layer": "rnn",
        }

        assert new_trials[1].params == {
            "/decoding_layer": "gru",
            "/encoding_layer": "gru",
        }


def test_duplicate_within_pool():
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate.
    """
    with create_producer() as (producer, storage):
        trials_in_db_before = len(storage._fetch_trials({}))
        new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

        # Avoid limiting number of samples from the within the algorithm.
        producer.experiment.algorithms.algorithm.pool_size = 1000

        producer.experiment.algorithms.algorithm.possible_values = [
            (v,) for v in [1, 1, 3]
        ]

        assert producer.produce(3) == 2

        # Algorithm was required to suggest some trials
        num_new_trials = producer.experiment.algorithms.algorithm._num
        assert num_new_trials == 3  # pool size

        # `num_new_trials` new trials were registered at database
        assert len(storage._fetch_trials({})) == trials_in_db_before + 2
        assert (
            len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 2
        )
        new_trials = list(storage._fetch_trials({"status": "new"}))
        assert new_trials[0].experiment == producer.experiment.id
        assert new_trials[0].start_time is None
        assert new_trials[0].end_time is None
        assert new_trials[0].results == []
        assert new_trials[0].params == {"x": 1}
        assert new_trials[1].params == {"x": 3}


def test_duplicate_within_pool_and_db():
    """Test that an algo suggesting multiple trials can have a few registered even
    if one of them is a duplicate with db.
    """
    with create_producer() as (producer, storage):
        trials_in_db_before = len(storage._fetch_trials({}))
        new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

        # Avoid limiting number of samples from the within the algorithm.
        producer.experiment.algorithms.algorithm.pool_size = 1000

        producer.experiment.algorithms.algorithm.possible_values = [
            (v,) for v in [0, 1, 2]
        ]

        assert producer.produce(3) == 1

        # Algorithm was required to suggest some trials
        num_new_trials = producer.experiment.algorithms.algorithm._num
        assert num_new_trials == 3  # pool size

        # `num_new_trials` new trials were registered at database
        assert len(storage._fetch_trials({})) == trials_in_db_before + 1
        assert (
            len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 1
        )
        new_trials = list(storage._fetch_trials({"status": "new"}))
        assert new_trials[0].experiment == producer.experiment.id
        assert new_trials[0].start_time is None
        assert new_trials[0].end_time is None
        assert new_trials[0].results == []
        assert new_trials[0].params == {"x": 1}


@pytest.mark.skip("Should be reactivated when algorithms can be warm-started")
def test_evc(monkeypatch, producer):
    """Verify that producer is using available trials from EVC"""
    experiment = producer.experiment
    new_experiment = build(
        experiment.name, algorithms="random", branching={"enable": True}
    )

    # Replace parent with hacked exp, otherwise parent ID does not match trials in DB
    # and fetch_trials() won't return anything.
    new_experiment._node.parent._item = experiment

    assert len(new_experiment.fetch_trials(with_evc_tree=True)) == len(
        experiment.fetch_trials()
    )

    producer.experiment = new_experiment

    def update_algo(trials):
        assert len(trials) == 7

    monkeypatch.setattr(producer, "_update_algorithm", update_algo)

    producer.update()


@pytest.mark.skip("Should be reactivated when algorithms can be warm-started")
def test_evc_duplicates(monkeypatch, producer):
    """Verify that producer won't register samples that are available in parent experiment"""
    experiment = producer.experiment
    new_experiment = build(
        experiment.name, algorithms="random", branching={"enable": True}
    )

    # Replace parent with hacked exp, otherwise parent ID does not match trials in DB
    # and fetch_trials() won't return anything.
    new_experiment._node.parent._item = experiment

    assert len(new_experiment.fetch_trials(with_evc_tree=True)) == len(
        experiment.fetch_trials()
    )

    trials = experiment.fetch_trials()

    def suggest(pool_size=None):
        suggest_trials = []
        while trials and len(suggest_trials) < pool_size:
            suggest_trials.append(trials.pop(0))
        return suggest_trials

    producer.experiment = new_experiment
    producer.algorithm = new_experiment.algorithms

    monkeypatch.setattr(new_experiment.algorithms, "suggest", suggest)

    producer.update()
    producer.produce(len(trials) + 2)

    assert len(trials) == 0
    assert len(new_experiment.fetch_trials(with_evc_tree=False)) == 0
