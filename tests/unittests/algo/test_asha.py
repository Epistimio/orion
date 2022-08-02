"""Tests for :mod:`orion.algo.asha`."""
from __future__ import annotations

import hashlib
import logging
from typing import ClassVar

import numpy as np
import pytest
from test_hyperband import (
    compare_registered_trial,
    create_rung_from_points,
    create_trial_for_hb,
    force_observe,
)

from orion.algo.asha import ASHA, ASHABracket, compute_budgets
from orion.algo.hyperband import RungDict
from orion.algo.space import Fidelity, Integer, Real, Space
from orion.core.worker.primary_algo import SpaceTransformAlgoWrapper
from orion.testing.algo import BaseAlgoTests, TestPhase
from orion.testing.trial import create_trial


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 9, 3))
    return space


@pytest.fixture
def b_config(space):
    """Return a configuration for a bracket."""
    fidelity_dim = space.values()[0]
    num_rungs = 3
    budgets = np.logspace(
        np.log(fidelity_dim.low) / np.log(fidelity_dim.base),
        np.log(fidelity_dim.high) / np.log(fidelity_dim.base),
        num_rungs,
        base=fidelity_dim.base,
    )
    return {
        "reduction_factor": fidelity_dim.base,
        "budgets": list(zip(list(map(int, budgets[::-1])), budgets)),
    }


@pytest.fixture
def asha(space: Space):
    """Return an instance of ASHA."""
    return ASHA(space)


@pytest.fixture
def bracket(b_config: dict, asha: ASHA):
    """Return a `ASHABracket` instance configured with `b_config`."""
    return ASHABracket(asha, b_config["budgets"], 1)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    return create_rung_from_points(np.linspace(0, 8, 9), n_trials=9, resources=1)


@pytest.fixture
def rung_1(rung_0: RungDict):
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_0["results"].values())[:3]]
    return create_rung_from_points(points, n_trials=3, resources=3)


@pytest.fixture
def rung_2(rung_1: RungDict):
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_1["results"].values())[:1]]
    return create_rung_from_points(points, n_trials=1, resources=9)


@pytest.fixture
def big_rung_0():
    """Create fake points and objectives for big rung 0."""
    n_rung_0 = 9 * 3
    return create_rung_from_points(
        np.linspace(0, n_rung_0 - 1, n_rung_0),
        n_trials=n_rung_0 * 2,
        resources=1,
    )


@pytest.fixture
def big_rung_1(big_rung_0: RungDict):
    """Create fake points and objectives for big rung 1."""
    n_rung_0 = len(big_rung_0["results"])
    n_rung_1 = 3 * 2
    return create_rung_from_points(
        np.linspace(n_rung_0 - n_rung_1, n_rung_0 - 1, n_rung_1),
        n_trials=n_rung_1 * 2,
        resources=3,
    )


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 16, 4, 3, 1) == [[(16, 1), (4, 4), (1, 16)]]
    # Check rounding (max_resources is not a multiple of reduction_factor)
    assert compute_budgets(1, 30, 4, 3, 1) == [[(16, 1), (4, 5), (1, 30)]]
    # Check rounding (min_resources may be rounded below its actual value)
    assert compute_budgets(25, 1000, 2, 6, 1) == [
        [(32, 25), (16, 52), (8, 109), (4, 229), (2, 478), (1, 1000)]
    ]
    # Check min_resources
    assert compute_budgets(5, 125, 5, 3, 1) == [[(25, 5), (5, 25), (1, 125)]]
    # Check num_rungs
    assert compute_budgets(1, 16, 2, 5, 1) == [
        [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]
    ]


def test_compute_compressed_budgets():
    """Verify proper computation of budgets when scale is small and integer rounding creates
    duplicates
    """
    assert compute_budgets(1, 16, 2, 10, 1) == [
        [
            (512, 1),
            (256, 2),
            (128, 3),
            (64, 4),
            (32, 5),
            (16, 6),
            (8, 7),
            (4, 9),
            (2, 12),
            (1, 16),
        ]
    ]

    with pytest.raises(ValueError) as exc:
        compute_budgets(1, 2, 2, 10, 1)

    assert "Cannot build budgets below max_resources" in str(exc.value)


class TestASHABracket:
    """Tests for the `ASHABracket` class."""

    def test_rungs_creation(self, bracket: ASHABracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0]["resources"] == 1
        assert bracket.rungs[1]["resources"] == 3
        assert bracket.rungs[2]["resources"] == 9

    def test_register(self, asha, bracket: ASHABracket):
        """Check that a point is correctly registered inside a bracket."""
        assert bracket.owner is asha
        trial = create_trial_for_hb((1, 0.0), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        bracket.register(trial)

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert trial.objective is not None
        assert bracket.rungs[0]["results"][trial_id][0] == trial.objective.value
        assert bracket.rungs[0]["results"][trial_id][1].to_dict() == trial.to_dict()

    def test_bad_register(self, asha: ASHA, bracket: ASHABracket):
        """Check that a non-valid point is not registered."""
        assert bracket.owner is asha

        with pytest.raises(IndexError) as ex:
            bracket.register(create_trial_for_hb((55, 0.0), 0.0))

        assert "Bad fidelity level 55" in str(ex.value)

    def test_candidate_promotion(
        self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict
    ):
        """Test that correct point is promoted."""
        assert bracket.owner is asha
        bracket.rungs[0] = rung_0

        point = bracket.get_candidates(0)[0]

        assert point.params == create_trial_for_hb((1, 0.0), 0.0).params

    def test_promotion_with_rung_1_hit(
        self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict
    ):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        trial = create_trial_for_hb((1, 0.0), None)
        assert bracket.owner is asha
        bracket.rungs[0] = rung_0
        assert trial.objective is not None
        bracket.rungs[1]["results"][asha.get_id(trial, ignore_fidelity=True)] = (
            trial.objective.value,
            trial,
        )

        trial = bracket.get_candidates(0)[0]

        assert trial.params == create_trial_for_hb((1, 1.0), 0.0).params

    def test_no_promotion_when_rung_full(
        self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict, rung_1: RungDict
    ):
        """Test that get_candidate returns `None` if rung 1 is full."""
        assert bracket.owner is asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        assert bracket.get_candidates(0) == []

    def test_no_promotion_if_not_enough_points(self, asha: ASHA, bracket: ASHABracket):
        """Test the get_candidate return None if there is not enough points ready."""
        assert bracket.owner is asha
        bracket.rungs[0] = RungDict(
            n_trials=1,
            resources=1,
            results={
                hashlib.md5(str([0.0]).encode("utf-8")).hexdigest(): (0.0, (1, 0.0))
            },
        )

        assert bracket.get_candidates(0) == []

    def test_no_promotion_if_not_completed(
        self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict
    ):
        """Test the get_candidate return None if trials are not completed."""
        assert bracket.owner is asha
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0]["results"]

        point = bracket.get_candidates(0)[0]

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        assert bracket.get_candidates(0) == []

    def test_is_done(self, bracket: ASHABracket, rung_0: RungDict):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2] = RungDict(n_trials=1, resources=9, results={"1": (1, 0.0)})

        assert bracket.is_done

    def test_update_rungs_return_candidate(
        self, asha: ASHA, bracket: ASHABracket, rung_1: RungDict
    ):
        """Check if a valid modified candidate is returned by update_rungs."""
        assert bracket.owner is asha
        bracket.rungs[1] = rung_1
        trial = create_trial_for_hb((3, 0.0), 0.0)

        candidate = bracket.promote(1)[0]

        trial_id = asha.get_id(trial, ignore_fidelity=True)
        assert trial_id in bracket.rungs[1]["results"]
        assert bracket.rungs[1]["results"][trial_id][1].params == trial.params
        assert candidate.params["epoch"] == 9

    def test_update_rungs_return_candidates(
        self,
        asha: ASHA,
        bracket: ASHABracket,
        big_rung_0: RungDict,
        big_rung_1: RungDict,
    ):
        """Check if many valid modified candidate is returned by update_rungs."""
        assert bracket.owner is asha

        bracket.rungs[0] = big_rung_0
        bracket.rungs[1] = big_rung_1

        candidates = bracket.promote(100)

        assert len(candidates) == 2 + 3 * 3
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 9)
            == 2
        )
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 3)
            == 3 * 3
        )

        candidates = bracket.promote(3)

        assert len(candidates) == 2 + 1
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 9)
            == 2
        )
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 3)
            == 1
        )

    def test_update_rungs_return_no_candidate(
        self, asha: ASHA, bracket: ASHABracket, rung_1: RungDict
    ):
        """Check if no candidate is returned by update_rungs."""
        assert bracket.owner is asha

        candidate = bracket.promote(1)

        assert candidate == []

    def test_repr(
        self, bracket: ASHABracket, rung_0: RungDict, rung_1: RungDict, rung_2: RungDict
    ):
        """Test the string representation of ASHABracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == "ASHABracket(resource=[1, 3, 9], repetition id=1)"


class TestASHA:
    """Tests for the algo ASHA."""

    def test_register(
        self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict, rung_1: RungDict
    ):
        """Check that a point is registered inside the bracket."""
        asha.brackets = [bracket]
        assert bracket.owner is asha
        bracket.rungs = [rung_0, rung_1]
        trial = create_trial_for_hb((1, 0.0), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        asha.observe([trial])

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert bracket.rungs[0]["results"][trial_id][0] == 0.0
        assert bracket.rungs[0]["results"][trial_id][1].params == trial.params

    def test_register_bracket_multi_fidelity(self, space: Space, b_config: dict):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 1
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        force_observe(asha, trial)

        bracket = asha.brackets[0]

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert bracket.rungs[0]["results"][trial_id][0] == 0.0
        assert bracket.rungs[0]["results"][trial_id][1].params == trial.params

        fidelity = 3
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        force_observe(asha, trial)

        assert len(bracket.rungs[1])
        assert trial_id in bracket.rungs[1]["results"]
        assert bracket.rungs[0]["results"][trial_id][1].params != trial.params
        assert bracket.rungs[1]["results"][trial_id][0] == 0.0
        assert bracket.rungs[1]["results"][trial_id][1].params == trial.params

    def test_register_next_bracket(self, space: Space, b_config: dict):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        force_observe(asha, trial)

        assert sum(len(rung["results"]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in asha.brackets[2].rungs) == 0
        assert trial_id in asha.brackets[1].rungs[0]["results"]
        compare_registered_trial(asha.brackets[1].rungs[0]["results"][trial_id], trial)

        value = 51
        fidelity = 9
        trial = create_trial_for_hb((fidelity, value), 0.0)
        trial_id = asha.get_id(trial, ignore_fidelity=True)

        force_observe(asha, trial)

        assert sum(len(rung["results"]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in asha.brackets[2].rungs) == 1
        assert trial_id in asha.brackets[2].rungs[0]["results"]
        compare_registered_trial(asha.brackets[2].rungs[0]["results"][trial_id], trial)

    def test_register_invalid_fidelity(self, space: Space, b_config: dict):
        """Check that a point cannot registered if fidelity is invalid."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 2
        trial = create_trial_for_hb((fidelity, value))

        asha.observe([trial])

        assert not asha.has_suggested(trial)
        assert not asha.has_observed(trial)

    def test_register_not_sampled(self, space: Space, b_config: dict, caplog):
        """Check that a point cannot registered if not sampled."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 2
        trial = create_trial_for_hb((fidelity, value))

        with caplog.at_level(logging.DEBUG, logger="orion.algo.hyperband"):
            asha.observe([trial])

        assert len(caplog.records) == 1
        assert "Ignoring trial" in caplog.records[0].msg

    def test_register_corrupted_db(self, caplog, space: Space, b_config: dict):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        trial = create_trial_for_hb((fidelity, value))

        force_observe(asha, trial)
        assert "Trial registered to wrong bracket" not in caplog.text

        fidelity = 1
        trial = create_trial_for_hb((fidelity, value), objective=0.0)

        caplog.clear()
        force_observe(asha, trial)
        assert "Trial registered to wrong bracket" in caplog.text

    def test_suggest_new(
        self,
        monkeypatch,
        asha: ASHA,
        bracket: ASHABracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that a new point is sampled."""
        asha.brackets = [bracket]
        assert bracket.owner is asha

        def sample(num=1, seed=None):
            return [create_trial_for_hb(("fidelity", 0.5))]

        monkeypatch.setattr(asha.space, "sample", sample)

        trials = asha.suggest(1)

        assert trials[0].params == {"epoch": 1, "lr": 0.5}

    def test_suggest_duplicates(
        self,
        monkeypatch,
        asha: ASHA,
        bracket: ASHABracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that sampling collisions are handled."""
        asha.brackets = [bracket]
        assert bracket.owner is asha

        fidelity = 1
        duplicate_trial = create_trial_for_hb((fidelity, 0.0))
        new_trial = create_trial_for_hb((fidelity, 0.5))

        duplicate_id_wo_fidelity = asha.get_id(duplicate_trial, ignore_fidelity=True)
        bracket.rungs[0] = dict(
            n_trials=2,
            resources=1,
            results={duplicate_id_wo_fidelity: (0.0, duplicate_trial)},
        )
        asha.trial_to_brackets[duplicate_id_wo_fidelity] = 0

        asha.register(duplicate_trial)

        trials = [duplicate_trial, new_trial]

        def sample(num=1, seed=None):
            return trials

        monkeypatch.setattr(asha.space, "sample", sample)

        assert asha.suggest(1)[0].params == new_trial.params

    def test_suggest_inf_duplicates(
        self,
        monkeypatch,
        asha: ASHA,
        bracket: ASHABracket,
        rung_0: RungDict,
        rung_1: RungDict,
        rung_2: RungDict,
    ):
        """Test that sampling inf collisions returns None."""
        asha.brackets = [bracket]
        assert bracket.owner is asha

        fidelity = 1
        zhe_trial = create_trial_for_hb((fidelity, 0.0))
        asha.trial_to_brackets[asha.get_id(zhe_trial, ignore_fidelity=True)] = 0

        def sample(num=1, seed=None):
            return [zhe_trial]

        monkeypatch.setattr(asha.space, "sample", sample)

        assert asha.suggest(1) == []

    def test_suggest_in_finite_cardinality(self):
        """Test that suggest None when search space is empty"""
        space = Space()
        space.register(Integer("yolo1", "uniform", 0, 5))
        space.register(Fidelity("epoch", 1, 9, 3))

        asha = ASHA(space)
        for i in range(6):
            force_observe(
                asha,
                create_trial(
                    (1, i),
                    names=("epoch", "yolo1"),
                    types=("fidelity", "integer"),
                    results={"objective": i},
                ),
            )

        for i in range(2):
            force_observe(
                asha,
                create_trial(
                    (3, i),
                    names=("epoch", "yolo1"),
                    types=("fidelity", "integer"),
                    results={"objective": i},
                ),
            )

        assert asha.suggest(1) == []

    def test_suggest_promote(self, asha: ASHA, bracket: ASHABracket, rung_0: RungDict):
        """Test that correct point is promoted and returned."""
        asha.brackets = [bracket]
        assert bracket.owner is asha
        bracket.rungs[0] = rung_0

        trials = asha.suggest(1)

        assert trials[0].params == {"epoch": 3, "lr": 0.0}

    def test_suggest_promote_many(
        self,
        asha: ASHA,
        bracket: ASHABracket,
        big_rung_0: RungDict,
        big_rung_1: RungDict,
    ):
        """Test that correct points are promoted and returned."""
        asha.brackets = [bracket]
        assert bracket.owner is asha
        bracket.rungs[0] = big_rung_0
        bracket.rungs[1] = big_rung_1

        candidates = asha.suggest(3)

        assert len(candidates) == 2 + 1
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 9)
            == 2
        )
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 3)
            == 1
        )

    def test_suggest_promote_many_plus_random(
        self,
        asha: ASHA,
        bracket: ASHABracket,
        big_rung_0: RungDict,
        big_rung_1: RungDict,
    ):
        """Test that correct points are promoted and returned, plus random points"""
        asha.brackets = [bracket]
        assert bracket.owner is asha
        bracket.rungs[0] = big_rung_0
        bracket.rungs[1] = big_rung_1

        candidates = asha.suggest(20)

        assert len(candidates) == 20
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 9)
            == 2
        )
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 3)
            == 3 * 3
        )
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 1)
            == 20 - 2 - 3 * 3
        )

    def test_suggest_promote_identic_objectives(
        self,
        asha: ASHA,
        bracket: ASHABracket,
        big_rung_0: RungDict,
        big_rung_1: RungDict,
    ):
        """Test that identic objectives are handled properly"""
        asha.brackets = [bracket]
        bracket.owner = asha

        n_trials = 9
        resources = 1

        results = {}
        for param in np.linspace(0, 8, 9):
            trial = create_trial_for_hb((resources, param), objective=0)
            trial_hash = trial.compute_trial_hash(
                trial,
                ignore_fidelity=True,
                ignore_experiment=True,
            )
            assert trial.objective is not None
            results[trial_hash] = (trial.objective.value, trial)

        bracket.rungs[0] = RungDict(
            n_trials=n_trials, resources=resources, results=results
        )

        candidates = asha.suggest(2)

        assert len(candidates) == 2
        assert (
            sum(1 for trial in candidates if trial.params[asha.fidelity_index] == 3)
            == 2
        )


BUDGETS = [
    16 + 8,  # rung 0
    (16 + 8 + 8 + 4),  # rung 1 (first bracket 8 4 2, second bracket 4)
    (16 + 8 + 4 + 8 + 4 + 2),  # rung 2
]


class TestGenericASHA(BaseAlgoTests):
    algo_name = "asha"
    config = {
        "seed": 123456,
        "num_rungs": 5,
        "num_brackets": 2,
        "repetitions": 3,
    }
    space = {
        "x": "uniform(0, 1, precision=15)",
        "y": "uniform(0, 1, precision=15)",
        "f": "fidelity(1, 10, base=2)",
    }

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        *[TestPhase(f"rung{i}", budget, "suggest") for i, budget in enumerate(BUDGETS)],
        TestPhase("rep1-rung1", BUDGETS[-1] + 16, "suggest"),
        TestPhase("rep2-rung1", BUDGETS[-1] * 2 + 16, "suggest"),
    ]
    _current_phase: TestPhase
    max_trials: ClassVar[int] = BUDGETS[-1] * 3

    def test_suggest_n(self):
        """Verify that suggest returns correct number of trials if ``num`` is specified in ``suggest``."""
        algo = self.create_algo()
        trials = algo.suggest(5)
        assert trials is not None
        if self._current_phase.name == "rung2":
            assert len(trials) == 3
        else:
            assert len(trials) == 5

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/598")
    def test_is_done_cardinality(self):
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(num_brackets=2, space=space)

        for rung in range(algo.algorithm.num_rungs):
            assert not algo.is_done

            trials = []
            while True:
                assert not algo.is_done
                n_sampled = len(algo.algorithm.sampled)
                n_trials = len(algo.algorithm.trial_to_brackets)
                new_trials = algo.suggest(1)
                if new_trials is None:
                    break
                trials += new_trials
                if rung == 0:
                    assert len(algo.algorithm.sampled) == n_sampled + 1
                else:
                    assert len(algo.algorithm.sampled) == n_sampled
                assert len(algo.algorithm.trial_to_brackets) == n_trials + 1

            assert not algo.is_done

            for i, trial in enumerate(trials):
                algo.observe([trial], [dict(objective=i)])

        assert algo.is_done

    def test_is_done_max_trials(self):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        rng = np.random.RandomState(123456)

        objective = 0
        while not algo.is_done:
            trials = algo.suggest(1)
            assert trials is not None
            if trials:
                self.observe_trials(trials, algo, rng)

        # ASHA should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def infer_repetition_and_rung(self, num: int):
        budgets = BUDGETS
        if num > budgets[-1] * 2:
            return 3, 0
        elif num > budgets[-1]:
            return 2, 0

        if num == 0:
            return 1, -1

        return 1, budgets.index(num)

    def assert_callbacks(self, spy, num: int, algo: SpaceTransformAlgoWrapper[ASHA]):

        if num == 0:
            return

        repetition_id, rung_id = self.infer_repetition_and_rung(num - 1)

        brackets = algo.algorithm.brackets
        assert brackets is not None
        assert len(brackets) == repetition_id * 2

        for j in range(0, rung_id + 1):
            for bracket in brackets:
                assert len(bracket.rungs[j]["results"]) > 0, (bracket, j)
