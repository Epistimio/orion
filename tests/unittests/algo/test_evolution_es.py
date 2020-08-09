#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.evolution_es`."""

import copy
import hashlib
import importlib

import numpy as np
import pytest

from orion.algo.evolution_es import BracketEVES, compute_budgets, EvolutionES
from orion.algo.space import Fidelity, Real, Space


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Fidelity('epoch', 1, 9, 1))
    return space


@pytest.fixture
def space1():
    """Create a Space with two real dimensions and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Real('weight_decay', 'uniform', 0, 1))
    space.register(Fidelity('epoch', 1, 8, 2))
    return space


@pytest.fixture
def space2():
    """Create a Space with two real dimensions."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Real('weight_decay', 'uniform', 0, 1))
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
    return BracketEVES(evolution, budgets, 1, space1)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 8, 9)
    return dict(
        n_trials=9,
        resources=1,
        results={hashlib.md5(str([point]).encode('utf-8')).hexdigest(): (point, (1, point))
                 for point in points})


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    values = map(lambda v: (v[0], (3, v[0])), list(sorted(rung_0['results'].values()))[:3])
    return dict(
        n_trials=3,
        resources=3,
        results={hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value
                 for value in values})


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 2."""
    values = map(lambda v: (v[0], (9, v[0])), list(sorted(rung_1['results'].values()))[:1])
    return dict(
        n_trials=1,
        resources=9,
        results={hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value
                 for value in values})


@pytest.fixture
def rung_3():
    """Create fake points and objectives for rung 3."""
    points = np.linspace(1, 4, 4)
    return dict(
        n_trials=4,
        resources=1,
        results={hashlib.md5(str([point]).encode('utf-8')).hexdigest():
                 (point, (np.power(2, (point - 1)), 1.0 / point, 1.0 / (point * point)))
                 for point in points})


@pytest.fixture
def rung_4():
    """Create duplicated fake points and objectives for rung 4."""
    points = np.linspace(1, 4, 4)
    return dict(
        n_trials=4,
        resources=1,
        results={hashlib.md5(str([point]).encode('utf-8')).hexdigest():
                 (point, (1, point // 2, point // 2))
                 for point in points})


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 3, 1, 2, 1) == [[(2, 1), (2, 2), (2, 3)]]
    assert compute_budgets(1, 4, 2, 4, 2) == [[(4, 1), (4, 2), (4, 4)]]


def test_customized_mutate_func(space2):
    """Verify customized mutate function works correctly"""
    org_data = [(1 / 2.0, 2.0), (1 / 8.0, 8.0), (1 / 5.0, 5.0), (1 / 4.0, 4.0)]
    mutated_data = []

    red_team = [(2.0, (0.5, 2.0)), (8.0, (0.13, 8.0))]
    blue_team = [(5.0, (0.2, 5.0)), (4.0, (0.25, 4.0))]

    for i, _ in enumerate(red_team):
        winner, loser = ((red_team, blue_team)
                         if red_team[i][0] < blue_team[i][0]
                         else (blue_team, red_team))

        mutated_data.append(winner[i][1])
        select_genes_key = i
        old = winner[i][1][select_genes_key]
        search_space = space2.values()[select_genes_key]

        mutate_attr = {}
        function_string = mutate_attr.pop('function',
                                          "orion.core.utils.tests.customized_mutate_example")
        mod_name, func_name = function_string.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        mutate_func = getattr(mod, func_name)

        new = mutate_func(search_space, old)
        if select_genes_key == 0:
            mutated_data.append((new, winner[i][1][1]))
        elif select_genes_key == 1:
            mutated_data.append((winner[i][1][0], new))

    assert len(mutated_data) == len(org_data)
    assert mutated_data[0] == org_data[0]
    assert mutated_data[2] == org_data[3]
    assert mutated_data[1][0] == org_data[0][0] / 2.0
    assert mutated_data[1][1] == org_data[0][1]
    assert mutated_data[3][0] == org_data[3][0]
    assert mutated_data[3][1] == org_data[3][1] / 2.0


class TestEvolutionES():
    """Tests for the algo Evolution."""

    def test_register(self, evolution, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        evolution.brackets = [bracket]
        bracket.hyperband = evolution
        bracket.eves = evolution
        bracket.rungs = [rung_0, rung_1]
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        evolution.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]['results']
        assert (0.0, point) == bracket.rungs[0]['results'][point_hash]


class TestBracketEVES():
    """Tests for `BracketEVES` class.."""

    def test_get_teams(self, bracket, rung_3):
        """Test that correct team is promoted."""
        bracket.rungs[0] = rung_3
        rung, population_range, red_team, blue_team = bracket._get_teams(0)
        assert len(list(rung.values())) == 4
        assert bracket.search_space_remove_fidelity == [1, 2]
        assert population_range == 4
        assert set(red_team).union(set(blue_team)) == {0, 1, 2, 3}
        assert set(red_team).intersection(set(blue_team)) == set()

    def test_mutate_population(self, bracket, rung_3):
        """Verify mutated candidates is generated correctly."""
        red_team = [0, 2]
        blue_team = [1, 3]
        population_range = 4
        for i in range(4):
            for j in [1, 2]:
                bracket.eves.population[j][i] = list(rung_3["results"].values())[i][1][j]
            bracket.eves.performance[i] = list(rung_3["results"].values())[i][0]

        org_data = np.stack((list(bracket.eves.population.values())[0],
                             list(bracket.eves.population.values())[1]), axis=0).T

        org_data = copy.deepcopy(org_data)

        bracket._mutate_population(red_team, blue_team,
                                   rung_3["results"], population_range)

        mutated_data = np.stack((list(bracket.eves.population.values())[0],
                                 list(bracket.eves.population.values())[1]), axis=0).T

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
            assert mutated_data[1][1] != org_data[2][1]

        if mutated_data[3][0] != org_data[2][0]:
            assert mutated_data[3][1] == org_data[2][1]
        else:
            assert mutated_data[3][1] != org_data[2][1]

    def test_duplicated_mutated_population(self, bracket, rung_4):
        """Verify duplicated candidates can be found and processed correctly."""
        red_team = [0, 2]
        blue_team = [0, 2]  # no mutate occur at first.
        population_range = 4
        for i in range(4):
            for j in [1, 2]:
                bracket.eves.population[j][i] = list(rung_4["results"].values())[i][1][j]
        points, nums_all_equal = bracket._mutate_population(red_team, blue_team,
                                                            rung_4["results"], population_range)

        # In this case, duplication will occur, and we can make it mutate one more time.
        # The points 1 and 2 should be different, while one of nums_all_equal should be 1.
        if points[1][1] != points[2][1]:
            assert points[1][2] == points[2][2]
        else:
            assert points[1][2] != points[2][2]

        assert nums_all_equal[0] == 0
        assert nums_all_equal[1] == 0
        assert nums_all_equal[2] == 1
        assert nums_all_equal[3] == 0

    def test_mutate_points(self, bracket, rung_3):
        """Test that correct point is promoted."""
        red_team = [0, 2]
        blue_team = [0, 2]
        population_range = 4
        for i in range(4):
            for j in [1, 2]:
                bracket.eves.population[j][i] = list(rung_3["results"].values())[i][1][j]
        points, nums_all_equal = bracket._mutate_population(red_team, blue_team,
                                                            rung_3["results"], population_range)
        assert points[0] == (1.0, 1.0, 1.0)
        assert points[1] == (2, 1.0 / 2, 1.0 / 4)
        assert (nums_all_equal == 0).all()
