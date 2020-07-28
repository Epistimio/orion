#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.evolution_es`."""

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
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Real('weight_decay', 'uniform', 0, 1))
    return space


@pytest.fixture
def budgets():
    """Return a configuration for a bracket."""
    return [(30, 4), (30, 5), (30, 6)]


@pytest.fixture
def evolution(space):
    """Return an instance of EvolutionES."""
    return EvolutionES(space, repetitions=1)


@pytest.fixture
def bracket(budgets, evolution, space):
    """Return a `Bracket` instance configured with `b_config`."""
    return BracketEVES(evolution, budgets, 1, space)


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
    """Create fake points and objectives for rung 1."""
    values = map(lambda v: (v[0], (9, v[0])), list(sorted(rung_1['results'].values()))[:1])
    return dict(
        n_trials=1,
        resources=9,
        results={hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value
                 for value in values})


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 3, 1, 2, 1) == [[(2, 1), (2, 2), (2, 3)]]
    assert compute_budgets(1, 4, 2, 4, 2) == [[(4, 1), (4, 2), (4, 4)]]


def test_get_mutated_candidates(space1):
    """Verify mutated candidates is generated correctly"""
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
        mod = importlib.import_module("orion.algo.mutate_functions")
        mutate_func = getattr(mod, "default_mutate")
        search_space = space1.values()[select_genes_key]
        new = mutate_func(search_space, old)
        if select_genes_key == 0:
            mutated_data.append((new, winner[i][1][1]))
        elif select_genes_key == 1:
            mutated_data.append((winner[i][1][0], new))

    assert len(mutated_data) == len(org_data)
    assert mutated_data[0] == org_data[0]
    assert mutated_data[2] == org_data[3]
    assert mutated_data[1][0] != org_data[0][0]
    assert mutated_data[1][1] == org_data[0][1]
    assert mutated_data[3][0] == org_data[3][0]
    assert mutated_data[3][1] != org_data[3][1]


def customized_mutate_example(search_space, old_value, **kwargs):
    """Define a customized mutate function example"""
    if search_space.type == "real":
        new_value = old_value / 2.0
    elif search_space.type == "integer":
        new_value = int(old_value + 1)
    else:
        new_value = old_value
    return new_value


def test_customized_mutate_func(space1):
    """Verify mutated candidates is generated correctly"""
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
        search_space = space1.values()[select_genes_key]
        new = customized_mutate_example(search_space, old)
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


def unchange_mutate(search_space, old_value, **kwargs):
    """Define an unchanged mutate example"""
    return old_value


def test_unchanged_mutate_cases(space1):
    """Verify mutated candidates is generated correctly"""
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
        search_space = space1.values()[select_genes_key]
        new = unchange_mutate(search_space, old)
        if select_genes_key == 0:
            mutated_data.append((new, winner[i][1][1]))
        elif select_genes_key == 1:
            mutated_data.append((winner[i][1][0], new))

    points = []
    for i in range(len(org_data)):
        point = [0] * 2
        point[0] = mutated_data[i][0]
        point[1] = mutated_data[i][1]
        nums_all_equal = 0
        while True:
            if tuple(point) in points:
                nums_all_equal += 1
                print("find equal one, continue to mutate.")
                select_genes_key = 0
                old = point[select_genes_key]
                search_space = space1.values()[select_genes_key]
                new = unchange_mutate(search_space, old)
                point[select_genes_key] = new
            else:
                break
            if nums_all_equal > 10:
                print("Can not Evolve any more, you can make an early stop.")
                break

        points.append(tuple(point))

    assert nums_all_equal == 11
    assert points == mutated_data


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
