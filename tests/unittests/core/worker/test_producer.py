#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import copy
import datetime
import time

import pytest

from orion.core.io.experiment_builder import build
from orion.core.utils import format_trials
from orion.core.utils.exceptions import ReservationTimeout, WaitingForTrials
from orion.core.worker.producer import Producer
from orion.core.worker.trial import Trial
from orion.testing.trial import compare_trials


def produce_lies(producer):
    """Wrap production of lies outside of `Producer.update`"""
    return producer._produce_lies(producer.experiment.fetch_noncompleted_trials())


def update_algorithm(producer):
    """Wrap update of algorithm outside of `Producer.update`"""
    return producer._update_algorithm(
        producer.experiment.fetch_trials_by_status("completed")
    )


@pytest.fixture()
def producer(monkeypatch, hacked_exp, random_dt, categorical_values):
    """Return a setup `Producer`."""
    # make init done

    possible_trials = [
        format_trials.tuple_to_trial(point, hacked_exp.space)
        for point in categorical_values
    ]
    hacked_exp.algorithms.algorithm.possible_values = possible_trials
    hacked_exp.algorithms.seed_rng(0)
    hacked_exp.max_trials = 20
    hacked_exp.algorithms.algorithm.max_trials = 20

    producer = Producer(hacked_exp)

    return producer


def test_algo_observe_completed(producer):
    """Test that algo only observes completed trials"""
    assert len(producer.experiment.fetch_trials()) > 3
    producer.update()
    # Algorithm must have received completed trials and their results
    obs_trials = producer.algorithm.algorithm._trials
    assert len(obs_trials) == 7
    assert obs_trials[0].params == {"/decoding_layer": "rnn", "/encoding_layer": "lstm"}
    assert obs_trials[1].params == {"/decoding_layer": "rnn", "/encoding_layer": "rnn"}
    assert obs_trials[2].params == {
        "/decoding_layer": "lstm_with_attention",
        "/encoding_layer": "gru",
    }
    assert obs_trials[0].objective.value == 3
    assert obs_trials[0].gradient is None
    assert obs_trials[0].constraints == []

    assert obs_trials[1].objective.value == 2
    assert obs_trials[1].gradient.value == [-0.1, 2]
    assert obs_trials[1].constraints == []

    assert obs_trials[2].objective.value == 10
    assert obs_trials[2].gradient.value == [5, 3]
    assert obs_trials[2].constraints[0].value == 1.2


def test_update_and_produce(producer, random_dt):
    """Test new trials are properly produced"""
    possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]
    producer.experiment.algorithms.algorithm.possible_values = possible_values

    producer.update()
    producer.produce(1)

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
    assert num_new_points == 1  # pool size

    compare_trials(producer.algorithm.algorithm._suggested, possible_values)


def test_register_new_trials(producer, storage, random_dt):
    """Test new trials are properly registered"""
    trials_in_db_before = len(storage._fetch_trials({}))
    new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

    producer.experiment.algorithms.algorithm.possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]

    producer.update()
    producer.produce(1)

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.algorithm.algorithm._num
    assert num_new_points == 1  # pool size

    # `num_new_points` new trials were registered at database
    assert len(storage._fetch_trials({})) == trials_in_db_before + 1
    assert len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 1
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


def test_concurent_producers(producer, storage, random_dt):
    """Test concurrent production of new trials."""
    trials_in_db_before = len(storage._fetch_trials({}))
    new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

    # Avoid limiting number of samples from the within the algorithm.
    producer.algorithm.algorithm.pool_size = 1000

    # Set so that first producer's algorithm generate valid trial on first time
    # And second producer produce same trial and thus must produce next one too.
    # Hence, we know that producer algo will have _num == 1 and
    # second producer algo will have _num == 2
    producer.algorithm.algorithm.possible_values = [
        format_trials.tuple_to_trial(point, producer.algorithm.space)
        for point in [("gru", "rnn"), ("gru", "gru")]
    ]
    # Make sure it starts from index 0
    producer.algorithm.seed_rng(0)

    second_producer = Producer(producer.experiment)
    second_producer.algorithm = copy.deepcopy(producer.algorithm)

    producer.update()
    second_producer.update()

    producer.produce(1)
    second_producer.produce(2)

    # Algorithm was required to suggest some trials
    num_new_trials = producer.algorithm.algorithm._num
    assert num_new_trials == 1  # pool size
    num_new_trials = second_producer.algorithm.algorithm._num
    assert num_new_trials == 2  # pool size

    # `num_new_trials` new trials were registered at database
    assert len(storage._fetch_trials({})) == trials_in_db_before + 2
    assert len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 2
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


def test_concurent_producers_shared_pool(producer, storage, random_dt):
    """Test concurrent production of new trials share the same pool"""
    trials_in_db_before = len(storage._fetch_trials({}))
    new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

    # Set so that first producer's algorithm generate valid trial on first time
    # And second producer produce same trial and thus must backoff and then stop
    # because first producer filled the pool.
    # Hence, we know that producer algo will have _num == 1 and
    # second producer algo will have _num == 1
    producer.algorithm.algorithm.possible_values = [
        format_trials.tuple_to_trial(point, producer.algorithm.space)
        for point in [("gru", "rnn"), ("gru", "gru")]
    ]
    # Make sure it starts from index 0
    producer.algorithm.seed_rng(0)

    second_producer = Producer(producer.experiment)
    second_producer.algorithm = copy.deepcopy(producer.algorithm)

    producer.update()
    second_producer.update()

    producer.produce(1)
    second_producer.produce(1)

    # Algorithm was required to suggest some trials
    num_new_trials = producer.algorithm.algorithm._num
    assert num_new_trials == 1  # pool size
    num_new_trials = second_producer.algorithm.algorithm._num
    assert num_new_trials == 1  # pool size

    # `num_new_trials` new trials were registered at database
    assert len(storage._fetch_trials({})) == trials_in_db_before + 1
    assert len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 1
    new_trials = list(
        storage._fetch_trials({"status": "new", "submit_time": random_dt})
    )
    assert len(new_trials) == 1
    assert new_trials[0].experiment == producer.experiment.id
    assert new_trials[0].start_time is None
    assert new_trials[0].end_time is None
    assert new_trials[0].results == []
    assert new_trials[0].params == {
        "/decoding_layer": "gru",
        "/encoding_layer": "rnn",
    }


def test_duplicate_within_pool(producer, storage, random_dt):
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate.
    """
    trials_in_db_before = len(storage._fetch_trials({}))
    new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

    # Avoid limiting number of samples from the within the algorithm.
    producer.algorithm.algorithm.pool_size = 1000

    producer.experiment.algorithms.algorithm.possible_values = [
        format_trials.tuple_to_trial(point, producer.algorithm.space)
        for point in [
            ("gru", "rnn"),
            ("gru", "rnn"),
            ("gru", "gru"),
        ]
    ]

    producer.update()
    producer.produce(3)

    # Algorithm was required to suggest some trials
    num_new_trials = producer.algorithm.algorithm._num
    assert num_new_trials == 3  # pool size

    # `num_new_trials` new trials were registered at database
    assert len(storage._fetch_trials({})) == trials_in_db_before + 2
    assert len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 2
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


def test_duplicate_within_pool_and_db(producer, storage, random_dt):
    """Test that an algo suggesting multiple trials can have a few registered even
    if one of them is a duplicate with db.
    """
    trials_in_db_before = len(storage._fetch_trials({}))
    new_trials_in_db_before = len(storage._fetch_trials({"status": "new"}))

    # Avoid limiting number of samples from the within the algorithm.
    producer.algorithm.algorithm.pool_size = 1000

    producer.experiment.algorithms.algorithm.possible_values = [
        format_trials.tuple_to_trial(point, producer.algorithm.space)
        for point in [
            ("gru", "rnn"),
            ("rnn", "rnn"),
            ("gru", "gru"),
        ]
    ]

    producer.update()
    producer.produce(3)

    # Algorithm was required to suggest some trials
    num_new_trials = producer.algorithm.algorithm._num
    assert num_new_trials == 3  # pool size

    # `num_new_trials` new trials were registered at database
    assert len(storage._fetch_trials({})) == trials_in_db_before + 2
    assert len(storage._fetch_trials({"status": "new"})) == new_trials_in_db_before + 2
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


def test_original_seeding(producer):
    """Verify that rng state in original algo changes when duplicate trials is discarded"""
    producer.algorithm.seed_rng(0)

    assert producer.algorithm.algorithm._index == 0

    producer.update()
    producer.produce(1)

    prev_index = producer.algorithm.algorithm._index
    prev_suggested = producer.algorithm.algorithm._suggested
    assert prev_index > 0

    # Force the algo back to 1 to make sure the RNG state of original algo keeps incrementing.
    producer.algorithm.seed_rng(0)

    producer.update()
    producer.produce(1)

    assert prev_suggested != producer.algorithm.algorithm._suggested
    assert prev_index < producer.algorithm.algorithm._index


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


def test_evc_duplicates(monkeypatch, producer):
    """Verify that producer wont register samples that are available in parent experiment"""
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


def test_suggest_n_max_trials(monkeypatch, producer):
    """Verify that producer suggest only max_trials - non_broken points."""
    producer.experiment.max_trials = 10
    producer.experiment.algorithms.algorithm.max_trials = 10
    producer = Producer(producer.experiment)

    def suggest_n(self, num):
        """Return duplicated points based on `num`"""
        return [
            format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
        ] * num

    monkeypatch.setattr(
        producer.experiment.algorithms.algorithm.__class__, "suggest", suggest_n
    )

    assert len(producer.experiment.fetch_trials(with_evc_tree=True)) == 7

    # Setup algorithm
    producer.update()

    assert producer.adjust_pool_size(50) == 3
    # Test pool_size is the min selected
    assert producer.adjust_pool_size(2) == 2
    producer.experiment.max_trials = 7
    assert producer.adjust_pool_size(50) == 1
    producer.experiment.max_trials = 5
    assert producer.adjust_pool_size(50) == 1

    trials = producer.experiment.fetch_trials()
    for trial in trials[:4]:
        producer.experiment._storage.set_trial_status(trial, "broken")

    assert len(producer.experiment.fetch_trials_by_status("broken")) == 4

    # Update broken count in producer
    producer.update()

    # There is now 3 completed and 4 broken. Max trials is 5. Producer should suggest 2
    assert producer.adjust_pool_size(50) == 2
    # Test pool_size is the min selected
    assert producer.adjust_pool_size(1) == 1
