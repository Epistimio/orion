#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.evolution_es`."""

import copy
import hashlib

import numpy as np
import pytest
from test_hyperband import (
    compare_registered_trial,
    create_rung_from_points,
    create_trial_for_hb,
    force_observe,
)

from orion.algo.evolution_es import BracketEVES, EvolutionES, compute_budgets
from orion.algo.space import Fidelity, Real, Space
from orion.testing.algo import BaseAlgoTests
from orion.testing.trial import create_trial


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 9, 1))
    return space


@pytest.fixture
def space1():
    """Create a Space with two real dimensions and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Real("weight_decay", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 8, 2))
    return space


@pytest.fixture
def space2():
    """Create a Space with two real dimensions."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Real("weight_decay", "uniform", 0, 1))
    return space


@pytest.fixture
def budgets():
    """Return a configuration for a bracket."""
    return [(30, 4), (30, 5), (30, 6)]


@pytest.fixture
def evolution(space1):
    """Return an instance of EvolutionES."""
    return EvolutionES(space1, repetitions=1, nums_population=4)


@pytest.fixture
def bracket(budgets, evolution, space1):
    """Return a `Bracket` instance configured with `b_config`."""
    return BracketEVES(evolution, budgets, 1)


@pytest.fixture
def evolution_customer_mutate(space1):
    """Return an instance of EvolutionES."""
    return EvolutionES(
        space1,
        repetitions=1,
        nums_population=4,
        mutate="orion.core.utils.tests.customized_mutate_example",
    )


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    return create_rung_from_points(np.linspace(0, 8, 9), n_trials=9, resources=1)


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_0["results"].values())[:3]]
    return create_rung_from_points(points, n_trials=3, resources=3)


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    points = [trial.params["lr"] for _, trial in sorted(rung_1["results"].values())[:1]]
    return create_rung_from_points(points, n_trials=1, resources=9)


@pytest.fixture
def rung_3(space1):
    """Create fake points and objectives for rung 3."""
    points = np.linspace(1, 4, 4)
    keys = list(space1.keys())
    types = [dim.type for dim in space1.values()]

    results = {}
    for point in points:
        trial = create_trial(
            (np.power(2, (point - 1)), 1.0 / point, 1.0 / (point * point)),
            names=keys,
            results={"objective": point},
            types=types,
        )
        trial_hash = trial.compute_trial_hash(
            trial,
            ignore_fidelity=True,
            ignore_experiment=True,
        )
        results[trial_hash] = (trial.objective.value, trial)

    return dict(
        n_trials=4,
        resources=1,
        results=results,
    )


@pytest.fixture
def rung_4(space1):
    """Create duplicated fake points and objectives for rung 4."""
    points = np.linspace(1, 4, 4)
    keys = list(space1.keys())
    types = [dim.type for dim in space1.values()]

    results = {}
    for point in points:
        trial = create_trial(
            (1, point // 2, point // 2),
            names=keys,
            results={"objective": point},
            types=types,
        )

        trial_hash = trial.compute_trial_hash(
            trial,
            ignore_fidelity=True,
            ignore_experiment=True,
        )
        results[trial_hash] = (trial.objective.value, trial)

    return dict(
        n_trials=4,
        resources=1,
        results=results,
    )


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 3, 1, 2, 1) == [[(2, 1), (2, 2), (2, 3)]]
    assert compute_budgets(1, 4, 2, 4, 2) == [[(4, 1), (4, 2), (4, 4)]]


def test_customized_mutate_population(space1, rung_3, budgets):
    """Verify customized mutated candidates is generated correctly."""
    customerized_dict = {
        "function": "orion.testing.algo.customized_mutate_example",
        "multiply_factor": 2.0,
        "add_factor": 1,
    }
    population_range = len(rung_3["results"])
    algo = EvolutionES(
        space1,
        repetitions=1,
        nums_population=population_range,
        mutate=customerized_dict,
    )
    algo.brackets[0] = BracketEVES(algo, budgets, 1)

    red_team = [0, 2]
    blue_team = [1, 3]
    rung_trials = list(rung_3["results"].values())
    for trial_index in range(population_range):
        objective, trial = rung_trials[trial_index]
        algo.performance[trial_index] = objective
        for ith_dim in [1, 2]:
            algo.population[ith_dim][trial_index] = trial.params[
                algo.space[ith_dim].name
            ]

    org_data = np.stack(
        (
            list(algo.brackets[0].eves.population.values())[0],
            list(algo.brackets[0].eves.population.values())[1],
        ),
        axis=0,
    ).T

    org_data = copy.deepcopy(org_data)

    algo.brackets[0]._mutate_population(
        red_team, blue_team, rung_3["results"], population_range, fidelity=2
    )

    mutated_data = np.stack(
        (
            list(algo.brackets[0].eves.population.values())[0],
            list(algo.brackets[0].eves.population.values())[1],
        ),
        axis=0,
    ).T

    # Winner team will be [0, 2], so [0, 2] will be remained, [1, 3] will be mutated.
    assert org_data.shape == mutated_data.shape
    assert (mutated_data[0] == org_data[0]).all()
    assert (mutated_data[2] == org_data[2]).all()
    assert (mutated_data[1] != org_data[1]).any()
    assert (mutated_data[3] != org_data[3]).any()
    assert (mutated_data[1] != org_data[0]).any()
    assert (mutated_data[3] != org_data[2]).any()

    # For each individual, mutation occurs in only one dimension chosen from two.
    # Customized test mutation function is divided by 2 for real type.
    if mutated_data[1][0] == org_data[0][0] / customerized_dict["multiply_factor"]:
        assert mutated_data[1][1] == org_data[0][1]
    else:
        assert (
            mutated_data[1][1] == org_data[0][1] / customerized_dict["multiply_factor"]
        )

    if mutated_data[3][0] == org_data[2][0] / customerized_dict["multiply_factor"]:
        assert mutated_data[3][1] == org_data[2][1]
    else:
        assert (
            mutated_data[3][1] == org_data[2][1] / customerized_dict["multiply_factor"]
        )


class TestEvolutionES:
    """Tests for the algo Evolution."""

    def test_register(self, evolution, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        evolution.brackets = [bracket]
        bracket.hyperband = evolution
        bracket.eves = evolution
        bracket.rungs = [rung_0, rung_1]
        trial = create_trial_for_hb((1, 0.0), objective=0.0)
        trial_id = evolution.get_id(trial, ignore_fidelity=True)

        evolution.observe([trial])

        assert len(bracket.rungs[0])
        assert trial_id in bracket.rungs[0]["results"]
        assert bracket.rungs[0]["results"][trial_id][0] == 0.0
        assert bracket.rungs[0]["results"][trial_id][1].params == trial.params


class TestBracketEVES:
    """Tests for `BracketEVES` class.."""

    def test_get_teams(self, bracket, rung_3):
        """Test that correct team is promoted."""
        bracket.rungs[0] = rung_3
        rung, population_range, red_team, blue_team = bracket._get_teams(0)
        assert len(list(rung.values())) == 4
        assert bracket.search_space_without_fidelity == [1, 2]
        assert population_range == 4
        assert set(red_team).union(set(blue_team)) == {0, 1, 2, 3}
        assert set(red_team).intersection(set(blue_team)) == set()

    def test_mutate_population(self, bracket, rung_3):
        """Verify mutated candidates is generated correctly."""
        red_team = [0, 2]
        blue_team = [1, 3]
        population_range = 4
        rung_trials = list(rung_3["results"].values())
        for trial_index in range(4):
            objective, trial = rung_trials[trial_index]

            bracket.eves.performance[trial_index] = objective
            for ith_dim in [1, 2]:
                bracket.eves.population[ith_dim][trial_index] = trial.params[
                    bracket.eves.space[ith_dim].name
                ]

        org_data = np.stack(
            (
                list(bracket.eves.population.values())[0],
                list(bracket.eves.population.values())[1],
            ),
            axis=0,
        ).T

        org_data = copy.deepcopy(org_data)

        bracket._mutate_population(
            red_team, blue_team, rung_3["results"], population_range, fidelity=2
        )

        mutated_data = np.stack(
            (
                list(bracket.eves.population.values())[0],
                list(bracket.eves.population.values())[1],
            ),
            axis=0,
        ).T

        # Winner team will be [0, 2], so [0, 2] will be remained, [1, 3] will be mutated.
        assert org_data.shape == mutated_data.shape
        assert (mutated_data[0] == org_data[0]).all()
        assert (mutated_data[2] == org_data[2]).all()
        assert (mutated_data[1] != org_data[1]).any()
        assert (mutated_data[3] != org_data[3]).any()
        assert (mutated_data[1] != org_data[0]).any()
        assert (mutated_data[3] != org_data[2]).any()

        # For each individual, mutation occurs in only one dimension chosen from two.
        if mutated_data[1][0] != org_data[0][0]:
            assert mutated_data[1][1] == org_data[0][1]
        else:
            assert mutated_data[1][1] != org_data[0][1]

        if mutated_data[3][0] != org_data[2][0]:
            assert mutated_data[3][1] == org_data[2][1]
        else:
            assert mutated_data[3][1] != org_data[2][1]

    def test_duplicated_mutated_population(self, bracket, rung_4):
        """Verify duplicated candidates can be found and processed correctly."""
        red_team = [0, 2]
        blue_team = [0, 2]  # no mutate occur at first.
        population_range = 4

        rung_trials = list(rung_4["results"].values())
        # Duplicate second item
        rung_trials.insert(2, rung_trials[1])
        for trial_index in range(4):
            objective, trial = rung_trials[trial_index]

            # bracket.eves.performance[trial_index] = objective
            for ith_dim in [1, 2]:
                bracket.eves.population[ith_dim][trial_index] = trial.params[
                    bracket.eves.space[ith_dim].name
                ]

        trials, nums_all_equal = bracket._mutate_population(
            red_team, blue_team, rung_4["results"], population_range, fidelity=2
        )

        # In this case, duplication will occur, and we can make it mutate one more time.
        # The trials 1 and 2 should be different, while one of nums_all_equal should be 1.
        if trials[1].params["lr"] != trials[2].params["lr"]:
            assert trials[1].params["weight_decay"] == trials[2].params["weight_decay"]
        else:
            assert trials[1].params["weight_decay"] != trials[2].params["weight_decay"]

        assert nums_all_equal[0] == 0
        assert nums_all_equal[1] == 0
        assert nums_all_equal[2] == 1
        assert nums_all_equal[3] == 0

    def test_mutate_trials(self, bracket, rung_3):
        """Test that correct trial is promoted."""
        red_team = [0, 2]
        blue_team = [0, 2]
        population_range = 4
        rung_trials = list(rung_3["results"].values())
        for trial_index in range(4):
            objective, trial = rung_trials[trial_index]

            # bracket.eves.performance[trial_index] = objective
            for ith_dim in [1, 2]:
                bracket.eves.population[ith_dim][trial_index] = trial.params[
                    bracket.eves.space[ith_dim].name
                ]

        trials, nums_all_equal = bracket._mutate_population(
            red_team, blue_team, rung_3["results"], population_range, fidelity=2
        )
        assert trials[0].params == {"epoch": 2, "lr": 1.0, "weight_decay": 1.0}
        assert trials[1].params == {"epoch": 2, "lr": 1.0 / 2, "weight_decay": 1.0 / 4}
        assert (nums_all_equal == 0).all()


class TestGenericEvolutionES(BaseAlgoTests):
    algo_name = "evolutiones"
    config = {
        "seed": 123456,
        "repetitions": 3,
        "nums_population": 20,
        "max_retries": 1000,
        "mutate": None,
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}

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

        for rung in range(len(algo.algorithm.brackets[0].rungs)):
            assert not algo.is_done
            n_sampled = len(algo.algorithm.sampled)
            n_trials = len(algo.algorithm.trial_to_brackets)
            trials = algo.suggest()
            if trials is None:
                break
            assert len(algo.algorithm.sampled) == n_sampled + len(trials)
            assert len(algo.algorithm.trial_to_brackets) == space.cardinality

            # We reached max number of trials we can suggest before observing any.
            assert algo.suggest() is None

            assert not algo.is_done

            for i, trial in enumerate(trials):
                backward.algo_observe(algo, [trial], [dict(objective=i)])

        assert algo.is_done

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        objective = 0
        while not algo.is_done:
            trials = algo.suggest(num)
            assert trials
            if trials:
                self.observe_trials(trials, algo, objective)
                objective += len(trials)

        # Hyperband should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def infer_repetition_and_rung(self, num):
        budgets = list(np.cumsum(BUDGETS))
        if num >= budgets[-1] * 2:
            return 3, -1
        elif num >= budgets[-1]:
            return 2, -1

        if num <= 1:
            return 1, -1

        return 1, budgets.index(num)

    def assert_callbacks(self, spy, num, algo):

        if num == 0:
            return

        repetition_id, rung_id = self.infer_repetition_and_rung(num - 1)

        brackets = algo.algorithm.brackets

        assert len(brackets) == repetition_id

        for j in range(0, rung_id + 1):
            for bracket in brackets:
                assert len(bracket.rungs[j]["results"]) > 0, (bracket, j)


BUDGETS = [20, 20, 20, 20]


TestGenericEvolutionES.set_phases(
    [("random", 0, "space.sample")]
    + [(f"rung{i}", budget, "suggest") for i, budget in enumerate(np.cumsum(BUDGETS))]
    + [("rep1-rung1", sum(BUDGETS), "suggest")]
    + [("rep2-rung1", sum(BUDGETS) * 2, "suggest")]
)
