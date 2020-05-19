#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""
import numpy
import pytest

from orion.algo.random import Random
from orion.algo.space import Integer, Real, Space


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'norm', 0.9)
    space.register(dim2)

    return space


def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    random_search = Random(space)

    random_search.seed_rng(1)
    a = random_search.suggest(1)[0]
    assert not numpy.allclose(a, random_search.suggest(1)[0])

    random_search.seed_rng(1)
    assert numpy.allclose(a, random_search.suggest(1)[0])


def test_set_state(space):
    """Verify that resetting state makes sampling deterministic"""
    random_search = Random(space)

    random_search.seed_rng(1)
    state = random_search.state_dict
    a = random_search.suggest(1)[0]
    assert not numpy.allclose(a, random_search.suggest(1)[0])

    random_search.set_state(state)
    assert numpy.allclose(a, random_search.suggest(1)[0])


def test_suggest_unique():
    """Verify that RandomSearch do not sample duplicates"""
    space = Space()
    space.register(Integer('yolo1', 'uniform', -3, 6))

    random_search = Random(space)

    n_samples = 6
    values = sum(random_search.suggest(n_samples), tuple())
    assert len(values) == n_samples
    assert len(set(values)) == n_samples


def test_suggest_unique_history():
    """Verify that RandomSearch do not sample duplicates based observed points"""
    space = Space()
    space.register(Integer('yolo1', 'uniform', -3, 6))

    random_search = Random(space)

    n_samples = 3
    values = sum(random_search.suggest(n_samples), tuple())
    assert len(values) == n_samples
    assert len(set(values)) == n_samples

    random_search.observe([[value] for value in values], [1] * n_samples)

    n_samples = 3
    new_values = sum(random_search.suggest(n_samples), tuple())
    assert len(new_values) == n_samples
    assert len(set(new_values)) == n_samples
    # No duplicates
    assert (set(new_values) & set(values)) == set()
