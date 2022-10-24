#!/usr/bin/env python
"""Tests for :mod:`orion.algo.hyperband`."""
from __future__ import annotations

import copy
import logging
import typing
from typing import Any, ClassVar, Sequence

import numpy as np
import pytest

from orion.algo.hyperband import (
    BudgetTuple,
    Hyperband,
    HyperbandBracket,
    RungDict,
    compute_budgets,
)
from orion.algo.space import Fidelity, Integer, Real, Space
from orion.core.utils.flatten import flatten
from orion.core.worker.primary_algo import SpaceTransform
from orion.core.worker.trial import Trial
from orion.testing.algo import BaseAlgoTests, TestPhase
from orion.testing.trial import compare_trials, create_trial

if typing.TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def create_trial_for_hb(point: Sequence[Any], objective: float | None = None) -> Trial:
    return create_trial(
        point,
        names=("epoch", "lr"),
        results={"objective": objective},
        types=("fidelity", "real"),
    )


def create_rung_from_points(
    points: Sequence[Any] | np.ndarray, n_trials: int, resources: int
) -> RungDict:

    results = {}
    for param in points:
        trial = create_trial_for_hb((resources, param), objective=param)
        trial_hash = trial.compute_trial_hash(
            trial,
            ignore_fidelity=True,
        )
        assert trial.objective is not None
        results[trial_hash] = (trial.objective.value, trial)

    return RungDict(n_trials=n_trials, resources=resources, results=results)


def compare_registered_trial(
    registered_trial: tuple[float | None, Trial], trial: Trial
) -> None:
    assert trial.objective is not None
    assert registered_trial[0] == trial.objective.value
    assert registered_trial[1].to_dict() == trial.to_dict()


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 9, 3))
    return space


@pytest.fixture
def budgets():
    """Return a configuration for a bracket."""
    return [BudgetTuple(9, 1), BudgetTuple(3, 3), BudgetTuple(1, 9)]


@pytest.fixture
def hyperband(space: Space):
    """Return an instance of Hyperband."""
    return Hyperband(space, repetitions=1)


@pytest.fixture
def bracket(budgets: list[BudgetTuple], hyperband: Hyperband):
    """Return a `HyperbandBracket` instance configured with `b_config`."""
    return HyperbandBracket(hyperband, budgets, 1)


@pytest.fixture
def rung_0() -> RungDict:
    """Create fake points and objectives for rung 0."""
    return create_rung_from_points(np.linspace(0, 8, 9), n_trials=9, resources=1)


@pytest.fixture
def rung_1(rung_0: RungDict) -> RungDict:
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_0["results"].values())[:3]]
    return create_rung_from_points(points, n_trials=3, resources=3)


@pytest.fixture
def rung_2(rung_1: RungDict) -> RungDict:
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_1["results"].values())[:1]]
    return create_rung_from_points(points, n_trials=1, resources=9)


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(81, 3) == [
        [(81, 1), (27, 3), (9, 9), (3, 27), (1, 81)],
        [(27, 3), (9, 9), (3, 27), (1, 81)],
        [(9, 9), (3, 27), (1, 81)],
        [(6, 27), (2, 81)],
        [(5, 81)],
    ]
    assert compute_budgets(16, 4) == [
        [(16, 1), (4, 4), (1, 16)],
        [(4, 4), (1, 16)],
        [(3, 16)],
    ]
    assert compute_budgets(16, 5) == [[(5, 3), (1, 16)], [(2, 16)]]


def force_observe(hyperband: Hyperband, trial: Trial) -> None:
    # hyperband.sampled.add(hashlib.md5(str(list(point)).encode("utf-8")).hexdigest())

    hyperband.register(trial)

    id_wo_fidelity = hyperband.get_id(trial, ignore_fidelity=True)

    bracket_index = hyperband.trial_to_brackets.get(id_wo_fidelity, None)

    if bracket_index is None:
        fidelity = flatten(trial.params)[hyperband.fidelity_index]
        assert hyperband.brackets is not None
        bracket_index = [
            i
            for i, bracket in enumerate(hyperband.brackets)
            if bracket.rungs[0]["resources"] == fidelity
        ][0]

    hyperband.trial_to_brackets[id_wo_fidelity] = bracket_index

    hyperband.observe([trial])


def mock_samples(hyperband: Hyperband, samples):
    assert hyperband.brackets is not None
    for bracket in hyperband.brackets:
        bracket._samples = samples


class TestHyperbandBracket:
    """Tests for the `HyperbandBracket` class."""

    def test_rungs_creation(self, bracket: HyperbandBracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0] == dict(n_trials=9, resources=1, results=dict())
        assert bracket.rungs[1] == dict(n_trials=3, resources=3, results=dict())
        assert bracket.rungs[2] == dict(n_trials=1, resources=9, results=dict())

    def test_register(self, hyperband: Hyperband, bracket: HyperbandBracket):
        """Check that a point is correctly registered inside a bracket."""
        assert bracket.owner is hyperband
        trial = create_trial_for_hb((1, 0.0), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        bracket.register(trial)

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert trial.objective is not None
        assert bracket.rungs[0]["results"][trial_id][0] == trial.objective.value
        assert bracket.rungs[0]["results"][trial_id][1].to_dict() == trial.to_dict()

    def test_bad_register(self, hyperband: Hyperband, bracket: HyperbandBracket):
        """Check that a non-valid point is not registered."""
        assert bracket.owner is hyperband

        with pytest.raises(IndexError) as ex:
            bracket.register(create_trial_for_hb((55, 0.0), 0.0))

        assert "Bad fidelity level 55" in str(ex.value)

    def test_candidate_promotion(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_0: RungDict
    ):
        """Test that correct point is promoted."""
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0

        points = bracket.get_candidates(0)

        assert points[0].params == create_trial_for_hb((1, 0.0), 0.0).params

    def test_promotion_with_rung_1_hit(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_0: RungDict
    ):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        trial = create_trial_for_hb((1, 0.0), None)
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0
        assert trial.objective is not None
        bracket.rungs[1]["results"][hyperband.get_id(trial, ignore_fidelity=True)] = (
            trial.objective.value,
            trial,
        )

        trials = bracket.get_candidates(0)

        assert trials[0].params == create_trial_for_hb((1, 1), 0.0).params

    def test_no_promotion_when_rung_full(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
    ):
        """Test that get_candidate returns `None` if rung 1 is full."""
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        points = bracket.get_candidates(0)

        assert points == []

    def test_no_promotion_if_not_completed(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_0: RungDict
    ):
        """Test the get_candidate return None if trials are not completed."""
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0]["results"]

        # points = bracket.get_candidates(0)

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        with pytest.raises(TypeError):
            bracket.get_candidates(0)

    def test_is_done(self, bracket: HyperbandBracket, rung_0: RungDict):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2]["results"] = {"1": (1, 0.0), "2": (1, 0.0), "3": (1, 0.0)}  # type: ignore

        assert bracket.is_done

    def test_update_rungs_return_candidate(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_1: RungDict
    ):
        """Check if a valid modified candidate is returned by update_rungs."""
        assert bracket.owner is hyperband
        bracket.rungs[1] = rung_1
        trial = create_trial_for_hb((3, 0.0), 0.0)

        candidates = bracket.promote(1)

        trial_id = hyperband.get_id(trial, ignore_fidelity=True)
        assert trial_id in bracket.rungs[1]["results"]
        assert bracket.rungs[1]["results"][trial_id][1].params == trial.params
        assert candidates[0].params["epoch"] == 9

    def test_update_rungs_return_no_candidate(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_1: RungDict
    ):
        """Check if no candidate is returned by update_rungs."""
        assert bracket.owner is hyperband

        candidates = bracket.promote(1)

        assert candidates == []

    def test_get_trial_max_resource(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test to get the max resource R for a particular trial"""
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0

        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 0.0))) == 1
        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 8.0))) == 1

        bracket.rungs[1] = rung_1
        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 0.0))) == 3
        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 8.0))) == 1

        bracket.rungs[2] = rung_2
        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 0.0))) == 9
        assert bracket.get_trial_max_resource(trial=create_trial_for_hb((1, 8.0))) == 1

    def test_repr(
        self,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test the string representation of HyperbandBracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == "HyperbandBracket(resource=[1, 3, 9], repetition id=1)"


class TestHyperband:
    """Tests for the algo Hyperband."""

    def test_register(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
    ):
        """Check that a point is registered inside the bracket."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband
        bracket.rungs = [rung_0, rung_1]
        trial = create_trial_for_hb((1, 0.0), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        hyperband.observe([trial])

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert bracket.rungs[0]["results"][trial_id][0] == 0.0
        assert bracket.rungs[0]["results"][trial_id][1].params == trial.params

    def test_register_bracket_multi_fidelity(self, space: Space):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 1
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        force_observe(hyperband, trial)
        assert hyperband.brackets is not None
        bracket = hyperband.brackets[0]

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert bracket.rungs[0]["results"][trial_id][0] == 0.0
        assert bracket.rungs[0]["results"][trial_id][1].params == trial.params

        fidelity = 3
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        force_observe(hyperband, trial)

        assert len(bracket.rungs[1])
        assert trial_id in bracket.rungs[1]["results"]
        assert bracket.rungs[0]["results"][trial_id][1].params != trial.params
        assert bracket.rungs[1]["results"][trial_id][0] == 0.0
        assert bracket.rungs[1]["results"][trial_id][1].params == trial.params

    def test_register_next_bracket(self, space: Space):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        force_observe(hyperband, trial)
        assert hyperband.brackets is not None
        assert sum(len(rung["results"]) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in hyperband.brackets[2].rungs) == 0
        assert trial_id in hyperband.brackets[1].rungs[0]["results"]
        compare_registered_trial(
            hyperband.brackets[1].rungs[0]["results"][trial_id], trial
        )

        value = 51
        fidelity = 9
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = hyperband.get_id(trial, ignore_fidelity=True)

        force_observe(hyperband, trial)

        assert sum(len(rung["results"]) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in hyperband.brackets[2].rungs) == 1
        assert trial_id in hyperband.brackets[2].rungs[0]["results"]
        compare_registered_trial(
            hyperband.brackets[2].rungs[0]["results"][trial_id], trial
        )

    def test_register_invalid_fidelity(self, space: Space):
        """Check that a point cannot registered if fidelity is invalid."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 2
        trial = create_trial_for_hb((fidelity, value))

        hyperband.observe([trial])

        assert not hyperband.has_suggested(trial)
        assert not hyperband.has_observed(trial)

    def test_register_not_sampled(self, space: Space, caplog: LogCaptureFixture):
        """Check that a point cannot registered if not sampled."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 2
        trial = create_trial_for_hb((fidelity, value))

        with caplog.at_level(logging.DEBUG, logger="orion.algo.hyperband"):
            hyperband.observe([trial])

        assert len(caplog.records) == 1
        assert "Ignoring trial" in caplog.records[0].msg

    def test_register_corrupted_db(self, caplog: LogCaptureFixture, space: Space):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        trial = create_trial_for_hb((fidelity, value))

        force_observe(hyperband, trial)
        assert "Trial registered to wrong bracket" not in caplog.text

        fidelity = 1
        trial = create_trial_for_hb((fidelity, value))

        caplog.clear()
        force_observe(hyperband, trial)
        assert "Trial registered to wrong bracket" in caplog.text

    def test_suggest_new(
        self,
        monkeypatch,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that a new point is sampled."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        mock_samples(
            hyperband, [create_trial_for_hb(("fidelity", i)) for i in range(10)]
        )

        trials = hyperband.suggest(100)
        assert trials is not None
        assert trials[0].params == {"epoch": 1.0, "lr": 0}
        assert trials[1].params == {"epoch": 1.0, "lr": 1}

    def test_suggest_duplicates_between_calls(
        self, monkeypatch, hyperband: Hyperband, bracket: HyperbandBracket
    ):
        """Test that same trials are not allowed in different suggest call of
        the same hyperband execution.
        """
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        fidelity = 1

        duplicate_trial = create_trial_for_hb((fidelity, 0.0))
        new_trial = create_trial_for_hb((fidelity, 0.5))

        duplicate_id = hyperband.get_id(duplicate_trial, ignore_fidelity=True)
        bracket.rungs[0]["results"] = {duplicate_id: (0.0, duplicate_trial)}  # type: ignore

        hyperband.trial_to_brackets[duplicate_id] = 0

        trials = [duplicate_trial, new_trial]

        mock_samples(
            hyperband,
            trials + [create_trial_for_hb((fidelity, i)) for i in range(10 - 2)],
        )
        trials = hyperband.suggest(100)
        assert trials is not None
        assert trials[0].params == new_trial.params

    def test_suggest_duplicates_one_call(
        self, monkeypatch, hyperband: Hyperband, bracket: HyperbandBracket
    ):
        """Test that same points are not allowed in the same suggest call ofxs
        the same hyperband execution.
        """
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        zhe_point = list(
            map(create_trial_for_hb, [(1, 0.0), (1, 1.0), (1, 1.0), (1, 2.0)])
        )

        mock_samples(hyperband, zhe_point * 2)
        zhe_samples = hyperband.suggest(100)
        assert zhe_samples is not None
        assert zhe_samples[0].params["lr"] == 0.0
        assert zhe_samples[1].params["lr"] == 1.0
        assert zhe_samples[2].params["lr"] == 2.0

        # zhe_point =
        mock_samples(
            hyperband,
            list(
                map(
                    create_trial_for_hb,
                    [
                        (3, 0.0),
                        (3, 1.0),
                        (3, 1.0),
                        (3, 2.0),
                        (3, 5.0),
                        (3, 4.0),
                    ],
                )
            ),
        )
        hyperband.trial_to_brackets[
            hyperband.get_id(create_trial_for_hb((1, 0.0)), ignore_fidelity=True)
        ] = 0
        hyperband.trial_to_brackets[
            hyperband.get_id(create_trial_for_hb((1, 0.0)), ignore_fidelity=True)
        ] = 0
        zhe_samples = hyperband.suggest(100)
        assert zhe_samples is not None
        assert zhe_samples[0].params["lr"] == 5.0
        assert zhe_samples[1].params["lr"] == 4.0

    def test_suggest_duplicates_between_execution(
        self, monkeypatch, hyperband: Hyperband, budgets: list[BudgetTuple]
    ):
        """Test that sampling collisions are handled between different hyperband execution."""
        hyperband.repetitions = 2
        bracket = HyperbandBracket(hyperband, budgets, 1)
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        for i in range(9):
            force_observe(hyperband, create_trial_for_hb((1, i), objective=i))

        for i in range(3):
            force_observe(hyperband, create_trial_for_hb((3, i), objective=i))

        force_observe(hyperband, create_trial_for_hb((9, 0), objective=0))

        assert not hyperband.is_done

        # lr:7 and lr:8 are already sampled in first repetition, they should not be present
        # in second repetition. Samples with lr:7 and lr:8 will be ignored.

        # (9, 0) already exists
        candidates_for_epoch_9_bracket = [(9, 0), (9, 2), (9, 3), (9, 10)]
        # (9, 1) -> (3, 1) already promoted in last repetition
        # (9, 3) sampled for previous bracket
        candidates_for_epoch_3_bracket = [(9, 1), (9, 3), (9, 4), (9, 5), (9, 11)]
        # (9, 0) -> (1, 0) already sampled in last repetition
        # (9, 8) -> (1, 8) already sampled in last repetition
        candidates_for_epoch_1_bracket = [(9, 0), (9, 8), (9, 12), (9, 13)]

        zhe_point = list(
            map(
                create_trial_for_hb,
                candidates_for_epoch_9_bracket
                + candidates_for_epoch_3_bracket
                + candidates_for_epoch_1_bracket,
            )
        )

        hyperband._refresh_brackets()
        mock_samples(hyperband, zhe_point)
        zhe_samples = hyperband.suggest(100)
        assert zhe_samples is not None
        assert len(zhe_samples) == 8
        assert zhe_samples[0].params == {"epoch": 9, "lr": 2}
        assert zhe_samples[1].params == {"epoch": 9, "lr": 3}
        assert zhe_samples[2].params == {"epoch": 9, "lr": 10}
        assert zhe_samples[3].params == {"epoch": 3, "lr": 4}
        assert zhe_samples[4].params == {"epoch": 3, "lr": 5}
        assert zhe_samples[5].params == {"epoch": 3, "lr": 11}
        assert zhe_samples[6].params == {"epoch": 1, "lr": 12}
        assert zhe_samples[7].params == {"epoch": 1, "lr": 13}

    def test_suggest_inf_duplicates(
        self,
        monkeypatch,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that sampling inf collisions will return None."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        zhe_trial = create_trial_for_hb(("fidelity", 0.0))
        hyperband.trial_to_brackets[
            hyperband.get_id(zhe_trial, ignore_fidelity=True)
        ] = 0

        mock_samples(hyperband, [zhe_trial] * 2)

        assert hyperband.suggest(100) == []

    def test_suggest_in_finite_cardinality(self):
        """Test that suggest None when search space is empty"""
        space = Space()
        space.register(Integer("yolo1", "uniform", 0, 5))
        space.register(Fidelity("epoch", 1, 9, 3))

        hyperband = Hyperband(space, repetitions=1)
        for i in range(6):
            force_observe(
                hyperband,
                create_trial(
                    (1, i),
                    names=("epoch", "yolo1"),
                    types=("fidelity", "integer"),
                    results={"objective": i},
                ),
            )

        assert hyperband.suggest(100) == []

    def test_suggest_promote(
        self, hyperband: Hyperband, bracket: HyperbandBracket, rung_0: RungDict
    ):
        """Test that correct point is promoted and returned."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0

        points = hyperband.suggest(100)
        assert points is not None
        assert len(points) == 3
        assert points[0].params == {"epoch": 3, "lr": 0}
        assert points[1].params == {"epoch": 3, "lr": 1}
        assert points[2].params == {"epoch": 3, "lr": 2}

    def test_suggest_promote_identic_objectives(
        self, hyperband: Hyperband, bracket: HyperbandBracket
    ):
        """Test that identic objectives are handled properly"""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        n_trials = 9
        resources = 1

        results = {}
        for param in np.linspace(0, 8, 9):
            trial = create_trial_for_hb((resources, param), objective=0)
            trial_hash = trial.compute_trial_hash(
                trial,
                ignore_fidelity=True,
            )
            assert trial.objective is not None
            results[trial_hash] = (trial.objective.value, trial)

        bracket.rungs[0] = RungDict(
            n_trials=n_trials, resources=resources, results=results
        )

        candidates = hyperband.suggest(2)
        assert candidates is not None
        assert len(candidates) == 2
        assert (
            sum(
                1 for trial in candidates if trial.params[hyperband.fidelity_index] == 3
            )
            == 2
        )

    def test_is_filled(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that Hyperband bracket detects when rung is filled."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert not bracket.is_filled
        assert not bracket.has_rung_filled(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert bracket.has_rung_filled(2)

    def test_is_ready(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that Hyperband bracket detects when rung is ready."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()
        assert not bracket.is_ready(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert bracket.is_ready(2)

    def test_suggest_opt_out(
        self,
        hyperband: Hyperband,
        bracket: HyperbandBracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that Hyperband opts out when rungs are not ready."""
        hyperband.brackets = [bracket]
        assert bracket.owner is hyperband

        bracket.rungs[0] = rung_0

        trial_id = next(iter(rung_0["results"].keys()))
        objective, point = rung_0["results"][trial_id]
        rung_0["results"][trial_id] = (None, point)

        assert hyperband.suggest(100) == []

    def test_full_process(self, monkeypatch, hyperband: Hyperband):
        """Test Hyperband full process."""
        sample_trials = [create_trial_for_hb(("fidelity", i)) for i in range(100)]

        hyperband._refresh_brackets()
        mock_samples(hyperband, copy.deepcopy(sample_trials))

        # Fill all brackets' first rung
        first_rung = hyperband.suggest(100)
        assert first_rung is not None
        first_bracket_first_rung = first_rung[6:]
        second_bracket_first_rung = first_rung[3:6]
        third_bracket_first_rung = first_rung[:3]

        compare_trials(
            first_bracket_first_rung,
            [create_trial_for_hb((1, i)) for i in range(6, 15)],
        )
        compare_trials(
            second_bracket_first_rung,
            [create_trial_for_hb((3, i)) for i in range(3, 6)],
        )
        compare_trials(
            third_bracket_first_rung, [create_trial_for_hb((9, i)) for i in range(3)]
        )
        assert hyperband.brackets is not None
        assert hyperband.brackets[0].has_rung_filled(0)
        assert not hyperband.brackets[0].is_ready()
        assert hyperband.suggest(100) == []
        assert hyperband.suggest(100) == []

        # Observe first bracket first rung
        for i, trial in enumerate(first_bracket_first_rung):
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=16 - i)
            )
        hyperband.observe(first_bracket_first_rung)

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote first bracket first rung
        first_bracket_second_rung = hyperband.suggest(100)
        compare_trials(
            first_bracket_second_rung,
            [create_trial_for_hb((3, 3 + 3 + 9 - 1 - i)) for i in range(3)],
        )

        assert hyperband.brackets[0].has_rung_filled(1)
        assert not hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        assert first_bracket_second_rung is not None
        # Observe first bracket second rung
        for i, trial in enumerate(first_bracket_second_rung):
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=8 - i)
            )
        hyperband.observe(first_bracket_second_rung)

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote first bracket second rung
        first_bracket_third_rung = hyperband.suggest(100)
        compare_trials(first_bracket_third_rung, [create_trial_for_hb((9, 12))])

        assert hyperband.brackets[0].has_rung_filled(2)
        assert not hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe second bracket first rung
        for i, trial in enumerate(second_bracket_first_rung):
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=8 - i)
            )
        hyperband.observe(second_bracket_first_rung)

        assert not hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote second bracket first rung
        second_bracket_second_rung = hyperband.suggest(100)
        compare_trials(second_bracket_second_rung, [create_trial_for_hb((9, 5))])

        assert not hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].has_rung_filled(1)
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe third bracket first rung
        for i, trial in enumerate(third_bracket_first_rung):
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=3 - i)
            )
        hyperband.observe(third_bracket_first_rung)

        assert not hyperband.brackets[0].is_ready(2)
        assert not hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[2].is_ready(0)
        assert hyperband.brackets[2].is_done

        assert second_bracket_second_rung is not None
        # Observe second bracket second rung
        for i, trial in enumerate(second_bracket_second_rung):
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=5 - i)
            )
        hyperband.observe(second_bracket_second_rung)

        assert not hyperband.brackets[0].is_ready(2)
        assert hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[1].is_done
        assert first_bracket_third_rung is not None
        hyperband.observe(first_bracket_third_rung)

        assert hyperband.is_done
        assert hyperband.brackets[0].is_done
        assert hyperband.suggest(100) == []

        # Refresh repeat and execution times property
        monkeypatch.setattr(hyperband, "repetitions", 2)
        # monkeypatch.setattr(hyperband.brackets[0], "repetition_id", 0)
        # hyperband.observe([(9, 12)], [{"objective": 3 - i}])
        assert len(hyperband.brackets) == 3
        hyperband._refresh_brackets()
        assert len(hyperband.brackets) == 6
        mock_samples(hyperband, copy.deepcopy(sample_trials[:3] + sample_trials))
        trials = hyperband.suggest(100)
        assert not hyperband.is_done
        assert not hyperband.brackets[3].is_ready(2)
        assert not hyperband.brackets[3].is_done
        assert trials is not None
        compare_trials(trials[:3], map(create_trial_for_hb, [(9, 3), (9, 4), (9, 6)]))
        compare_trials(trials[3:6], map(create_trial_for_hb, [(3, 7), (3, 8), (3, 9)]))
        compare_trials(trials[6:], [create_trial_for_hb((1, i)) for i in range(15, 24)])


BUDGETS = [20, 8, 3, 1]


class TestGenericHyperband(BaseAlgoTests):
    algo_name = "hyperband"
    config = {
        "seed": 123456,
        "repetitions": 3,
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        *[
            TestPhase(f"rung{i}", budget, "suggest")
            for i, budget in enumerate(np.cumsum(BUDGETS[:-1]))
        ],
        TestPhase("rep1-rung1", sum(BUDGETS), "suggest"),
        TestPhase("rep2-rung1", sum(BUDGETS) * 2, "suggest"),
    ]

    def test_suggest_lots(self):
        """Test that hyperband returns whole rungs when requesting large `num`"""
        algo = self.create_algo()
        num = algo.n_observed
        points = algo.suggest(10000)
        repetition_id, rung_id = self.infer_repetition_and_rung(num)
        assert points is not None
        assert len(points) == BUDGETS[rung_id + 1 if rung_id < 3 else 0], (
            BUDGETS,
            rung_id,
            repetition_id,
            num,
        )

    def test_suggest_n(self):
        algo = self.create_algo()
        num = algo.n_observed
        points = algo.suggest(5)
        repetition_id, rung_id = self.infer_repetition_and_rung(num)
        assert points is not None
        assert len(points) == min(BUDGETS[rung_id + 1 if rung_id < 3 else 0], 5)

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/598")
    def test_is_done_cardinality(self):
        space = self.update_space(
            {
                # Increase fidelity to increase number of trials in first rungs
                "f": "fidelity(1, 100, base=2)",
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        assert isinstance(algo.algorithm, Hyperband)
        assert algo.algorithm.brackets is not None
        for rung in range(len(algo.algorithm.brackets[0].rungs)):
            assert not algo.is_done

            trials: list[Trial] = []
            assert not algo.is_done
            n_sampled = algo.n_suggested
            n_trials = len(algo.algorithm.trial_to_brackets)
            new_trials = algo.suggest(100)
            if not new_trials:
                break
            trials += new_trials
            assert algo.n_suggested == n_sampled + len(new_trials)
            assert len(algo.algorithm.trial_to_brackets) == space.cardinality

            # We reached max number of trials we can suggest before observing any.
            assert algo.suggest(100) == []

            assert not algo.is_done

            for i, trial in enumerate(trials):
                trial.results = [dict(objective=i)]
                algo.observe([trial])

        assert algo.is_done

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num: int):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        rng = np.random.RandomState(123456)

        while not algo.is_done:
            trials = algo.suggest(num)
            assert trials is not None
            if trials:
                self.observe_trials(trials, algo, rng)

        # Hyperband should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def infer_repetition_and_rung(self, num: int) -> tuple[int, int]:
        budgets: list[int] = list(np.cumsum(BUDGETS))
        if num >= budgets[-1] * 2:
            return 3, -1
        elif num >= budgets[-1]:
            return 2, -1

        if num <= 1:
            return 1, -1

        return 1, budgets.index(num)

    def assert_callbacks(self, spy, num: int, algo: SpaceTransform[Hyperband]):
        if num == 0:
            return

        repetition_id, rung_id = self.infer_repetition_and_rung(num - 1)

        brackets = algo.algorithm.brackets
        assert brackets is not None
        assert len(brackets) == repetition_id * len(brackets[0].rungs)

        for j in range(0, rung_id + 1):
            for bracket in brackets:
                if len(bracket.rungs) > j:
                    assert len(bracket.rungs[j]["results"]) > 0, (bracket, j)
