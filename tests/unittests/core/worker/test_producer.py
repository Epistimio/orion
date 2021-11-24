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


class DumbParallelStrategy:
    """Mock object for parallel strategy"""

    def observe(self, trials):
        """See ParallelStrategy.observe"""
        self._observed_trials = trials
        self._value = None

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        if self._value:
            value = self._value
        else:
            value = len(self._observed_trials)

        self._lie = lie = Trial.Result(name="lie", type="lie", value=value)
        return lie


def produce_lies(producer):
    """Wrap production of lies outside of `Producer.update`"""
    return producer._produce_lies(producer.experiment.fetch_noncompleted_trials())


def update_algorithm(producer):
    """Wrap update of algorithm outside of `Producer.update`"""
    return producer._update_algorithm(
        producer.experiment.fetch_trials_by_status("completed")
    )


def update_naive_algorithm(producer):
    """Wrap update of naive algorithm outside of `Producer.update`"""
    return producer._update_naive_algorithm(
        producer.experiment.fetch_noncompleted_trials()
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

    hacked_exp.producer["strategy"] = DumbParallelStrategy()

    producer = Producer(hacked_exp)

    return producer


def test_algo_observe_completed(producer):
    """Test that algo only observes completed trials"""
    assert len(producer.experiment.fetch_trials()) > 3
    producer.update()
    # Algorithm must have received completed trials and their results
    obs_trials = producer.algorithm.algorithm._trials
    assert len(obs_trials) == 3
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


def test_strategist_observe_completed(producer):
    """Test that strategist only observes completed trials"""
    assert len(producer.experiment.fetch_trials()) > 3
    producer.update()
    # Algorithm must have received completed points and their results
    obs_trials = producer.strategy._observed_trials
    assert len(obs_trials) == 3
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


def test_naive_algorithm_is_producing(monkeypatch, producer, random_dt):
    """Verify naive algo is used to produce, not original algo"""
    producer.algorithm.algorithm.possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]
    producer.update()
    monkeypatch.setattr(producer.algorithm.algorithm, "set_state", lambda value: None)
    producer.algorithm.algorithm.possible_values = [
        format_trials.tuple_to_trial(("gru", "gru"), producer.algorithm.space)
    ]
    producer.produce(1)

    assert producer.naive_algorithm.algorithm._num == 1  # pool size
    assert producer.algorithm.algorithm._num == 0


def test_update_and_produce(producer, random_dt):
    """Test new trials are properly produced"""
    possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]
    producer.experiment.algorithms.algorithm.possible_values = possible_values

    producer.update()
    producer.produce(1)

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.naive_algorithm.algorithm._num
    assert num_new_points == 1  # pool size

    compare_trials(producer.naive_algorithm.algorithm._suggested, possible_values)


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
    num_new_points = producer.naive_algorithm.algorithm._num
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


def test_no_lies_if_all_trials_completed(producer, storage, random_dt):
    """Verify that no lies are created if all trials are completed"""
    query = {"status": {"$ne": "completed"}}
    storage.delete_trials(producer.experiment, where=query)
    trials_in_db_before = len(storage.fetch_trials(experiment=producer.experiment))
    assert trials_in_db_before == 3

    producer.update()

    assert len(produce_lies(producer)) == 0


def test_lies_generation(producer, storage, random_dt):
    """Verify that lies are created properly"""
    query = {"status": {"$ne": "completed"}}
    trials_non_completed = storage.fetch_trials(
        experiment=producer.experiment, where=query
    )
    assert len(trials_non_completed) == 4
    query = {"status": "completed"}
    trials_completed = storage.fetch_trials(experiment=producer.experiment, where=query)
    assert len(trials_completed) == 3

    producer.update()

    lies = produce_lies(producer)
    assert len(lies) == 4

    trials_non_completed = list(
        sorted(
            trials_non_completed,
            key=lambda trial: trial.submit_time,
        )
    )

    for i in range(4):
        trials_non_completed[i]._id = lies[i].id
        trials_non_completed[i].status = "completed"
        trials_non_completed[i].end_time = random_dt
        trials_non_completed[i].results.append(producer.strategy._lie)
        trials_non_completed[i].parents = set([trial.id for trial in trials_completed])
        lies_dict = lies[i].to_dict()
        lies_dict["parents"] = set(lies_dict["parents"])
        assert lies_dict == trials_non_completed[i].to_dict()


def test_register_lies(producer, storage, random_dt):
    """Verify that lies are registed in DB properly"""
    query = {"status": {"$ne": "completed"}}
    trials_non_completed = list(
        storage.fetch_trials(experiment=producer.experiment, where=query)
    )
    assert len(trials_non_completed) == 4
    query = {"status": "completed"}
    trials_completed = list(
        storage.fetch_trials(experiment=producer.experiment, where=query)
    )
    assert len(trials_completed) == 3

    producer.update()
    produce_lies(producer)

    lying_trials = storage._db.read("lying_trials")
    assert len(lying_trials) == 4

    trials_non_completed = list(
        sorted(
            trials_non_completed,
            key=lambda trial: trial.submit_time,
        )
    )

    for i in range(4):
        trials_non_completed[i]._id = lying_trials[i]["_id"]
        trials_non_completed[i].status = "completed"
        trials_non_completed[i].end_time = random_dt
        trials_non_completed[i].results.append(producer.strategy._lie)
        trials_non_completed[i].parents = set([trial.id for trial in trials_completed])
        lying_trials[i]["parents"] = set(lying_trials[i]["parents"])
        assert lying_trials[i] == trials_non_completed[i].to_dict()


def test_register_duplicate_lies(producer, storage, random_dt):
    """Verify that duplicate lies are not registered twice in DB"""
    query = {"status": {"$ne": "completed"}}
    trials_non_completed = storage.fetch_trials(
        experiment=producer.experiment, where=query
    )
    assert len(trials_non_completed) == 4

    # Overwrite value of lying result of the strategist so that all lying trials have the same value
    # otherwise they would not be exact duplicates.
    producer.strategy._value = 4

    # Set specific output value for to algo to ensure successful creation of a new trial.
    producer.experiment.algorithms.algorithm.possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]

    producer.update()
    lies = produce_lies(producer)
    assert len(lies) == 4
    lying_trials = list(storage._db.read("lying_trials"))
    assert len(lying_trials) == 4

    # Create a new point to make sure additional non-completed trials increase number of lying
    # trials generated
    producer.produce(1)

    trials_non_completed = storage._fetch_trials(query)
    assert len(trials_non_completed) == 5

    producer.update()

    assert len(produce_lies(producer)) == 5
    lying_trials = list(storage._db.read("lying_trials"))
    assert len(lying_trials) == 5

    # Make sure trying to generate again does not add more fake trials since they are identical
    assert len(produce_lies(producer)) == 5
    lying_trials = list(storage._db.read("lying_trials"))
    assert len(lying_trials) == 5


def test_register_duplicate_lies_with_different_results(producer, storage, random_dt):
    """Verify that duplicate lies with different results are all registered in DB"""
    query = {"status": {"$ne": "completed"}, "experiment": producer.experiment.id}
    trials_non_completed = list(storage._fetch_trials(query))
    assert len(trials_non_completed) == 4

    # Overwrite value of lying result to force different results.
    producer.strategy._value = 11

    assert len(produce_lies(producer)) == 4
    lying_trials = storage._db.read("lying_trials")
    assert len(lying_trials) == 4

    # Overwrite value of lying result to force different results.
    producer.strategy._value = new_lying_value = 12

    lying_trials = produce_lies(producer)
    assert len(lying_trials) == 4
    nb_lying_trials = len(storage._db.read("lying_trials"))
    assert nb_lying_trials == 4 + 4
    assert lying_trials[0].lie.value == new_lying_value


def test_naive_algo_not_trained_when_all_trials_completed(producer, storage, random_dt):
    """Verify that naive algo is not trained on additional trials when all completed"""
    query = {"status": {"$ne": "completed"}}
    storage.delete_trials(producer.experiment, where=query)
    trials_in_db_before = len(storage.fetch_trials(producer.experiment))
    assert trials_in_db_before == 3

    producer.update()

    assert len(producer.algorithm.algorithm._trials) == 3
    assert len(producer.naive_algorithm.algorithm._trials) == 3


def test_naive_algo_trained_on_all_non_completed_trials(producer, storage, random_dt):
    """Verify that naive algo is trained on additional trials"""
    # Set two of completed trials to broken and reserved to have all possible status
    query = {"experiment": producer.experiment.id, "status": "completed"}
    completed_trials = storage._fetch_trials(query)

    storage.set_trial_status(completed_trials[0], "broken")
    storage.set_trial_status(completed_trials[1], "reserved")

    # Make sure non completed trials and completed trials are set properly for the unit-test
    query = {"status": {"$ne": "completed"}, "experiment": producer.experiment.id}
    non_completed_trials = storage._fetch_trials(query)
    assert len(non_completed_trials) == 6
    # Make sure we have all type of status except completed
    assert set(trial.status for trial in non_completed_trials) == set(
        ["new", "reserved", "suspended", "interrupted", "broken"]
    )
    query = {"status": "completed", "experiment": producer.experiment.id}
    assert len(storage._fetch_trials(query)) == 1

    # Executing the actual test
    producer.update()
    assert len(produce_lies(producer)) == 6

    assert len(producer.algorithm.algorithm._trials) == 1
    assert len(producer.naive_algorithm.algorithm._trials) == (1 + 6)


def test_naive_algo_is_discared(producer, monkeypatch):
    """Verify that naive algo is discarded and recopied from original algo"""
    # Set values for predictions
    producer.experiment.algorithms.algorithm.possible_values = [
        format_trials.tuple_to_trial(("gru", "rnn"), producer.algorithm.space)
    ]

    producer.update()
    assert len(produce_lies(producer)) == 4

    first_naive_algorithm = producer.naive_algorithm

    assert len(producer.algorithm.algorithm._trials) == 3
    assert len(first_naive_algorithm.algorithm._trials) == (3 + 4)

    producer.produce(1)

    # Only update the original algo, naive algo is still not discarded
    update_algorithm(producer)
    assert len(producer.algorithm.algorithm._trials) == 3
    assert first_naive_algorithm == producer.naive_algorithm
    assert len(producer.naive_algorithm.algorithm._trials) == (3 + 4)

    # Discard naive algo and create a new one, now trained on 5 trials.
    update_naive_algorithm(producer)
    assert first_naive_algorithm != producer.naive_algorithm
    assert len(producer.naive_algorithm.algorithm._trials) == (3 + 5)


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
    # This is necessary because naive_algo is recopied from original algo and thus would always get
    # the same RNG state if the original algo RNG state would not increment.
    # See `Producer.produce` to observe the dummy `self.algorith.suggest()` used to increment
    # original algo's RNG state.
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
        assert len(trials) == 3

    def update_naive_algo(trials):
        assert len(trials) == 4

    monkeypatch.setattr(producer, "_update_algorithm", update_algo)
    monkeypatch.setattr(producer, "_update_naive_algorithm", update_naive_algo)

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

    # Setup naive algorithm
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
