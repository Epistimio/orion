#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.evolution_es`."""

import hashlib

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


class TestEvolutionES():
    """Tests for the algo Hyperband."""

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
