#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.asha`."""

import hashlib

import numpy as np
import pytest

from orion.algo.asha import ASHA, Bracket, compute_budgets
from orion.algo.space import Fidelity, Integer, Real, Space


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Fidelity('epoch', 1, 9, 3))
    return space


@pytest.fixture
def b_config(space):
    """Return a configuration for a bracket."""
    fidelity_dim = space.values()[0]
    num_rungs = 3
    budgets = np.logspace(
        np.log(fidelity_dim.low) / np.log(fidelity_dim.base),
        np.log(fidelity_dim.high) / np.log(fidelity_dim.base),
        num_rungs, base=fidelity_dim.base)
    return {'reduction_factor': fidelity_dim.base, 'budgets': budgets}


@pytest.fixture
def asha(b_config, space):
    """Return an instance of ASHA."""
    return ASHA(space)


@pytest.fixture
def bracket(b_config):
    """Return a `Bracket` instance configured with `b_config`."""
    return Bracket(None, b_config['reduction_factor'], b_config['budgets'])


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 1, 9)
    return (1, {hashlib.md5(str([point]).encode('utf-8')).hexdigest():
            (point, (1, point)) for point in points})


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    return (3, {hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value for value in
            map(lambda v: (v[0], (3, v[0])), sorted(rung_0[1].values()))})


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    return (9, {hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value for value in
            map(lambda v: (v[0], (9, v[0])), sorted(rung_1[1].values()))})


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 16, 4, 3) == [1, 4, 16]
    # Check rounding (max_resources is not a multiple of reduction_factor)
    assert compute_budgets(1, 30, 4, 3) == [1, 5, 30]
    # Check rounding (min_resources may be rounded below its actual value)
    assert compute_budgets(25, 1000, 2, 6) == [25, 52, 109, 229, 478, 1000]
    # Check min_resources
    assert compute_budgets(5, 125, 5, 3) == [5, 25, 125]
    # Check num_rungs
    assert compute_budgets(1, 16, 2, 5) == [1, 2, 4, 8, 16]


def test_compute_compressed_budgets():
    """Verify proper computation of budgets when scale is small and integer rounding creates
    duplicates
    """
    assert compute_budgets(1, 16, 2, 10) == [1, 2, 3, 4, 5, 6, 7, 9, 12, 16]

    with pytest.raises(ValueError) as exc:
        compute_budgets(1, 2, 2, 10)

    assert 'Cannot build budgets below max_resources' in str(exc.value)


class TestBracket():
    """Tests for the `Bracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0][0] == 1
        assert bracket.rungs[1][0] == 3
        assert bracket.rungs[2][0] == 9

    def test_register(self, asha, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.asha = asha
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

    def test_bad_register(self, asha, bracket):
        """Check that a non-valid point is not registered."""
        bracket.asha = asha

        with pytest.raises(IndexError) as ex:
            bracket.register((55, 0.0), 0.0)

        assert 'Bad fidelity level 55' in str(ex.value)

    def test_candidate_promotion(self, asha, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        point = bracket.get_candidate(0)

        assert point == (1, 0.0)

    def test_promotion_with_rung_1_hit(self, asha, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1][1][point_hash] = (0.0, point)

        point = bracket.get_candidate(0)

        assert point == (1, 0.125)

    def test_no_promotion_when_rung_full(self, asha, bracket, rung_0, rung_1):
        """Test that get_candidate returns `None` if rung 1 is full."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        point = bracket.get_candidate(0)

        assert point is None

    def test_no_promotion_if_not_enough_points(self, asha, bracket):
        """Test the get_candidate return None if there is not enough points ready."""
        bracket.asha = asha
        bracket.rungs[0] = (1, {hashlib.md5(str([0.0]).encode('utf-8')).hexdigest():
                                (0.0, (1, 0.0))})

        point = bracket.get_candidate(0)

        assert point is None

    def test_no_promotion_if_not_completed(self, asha, bracket, rung_0):
        """Test the get_candidate return None if trials are not completed."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0][1]

        point = bracket.get_candidate(0)

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        point = bracket.get_candidate(0)

        assert point is None

    def test_is_done(self, bracket, rung_0):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2] = (9, {'1': (1, 0.0)})

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, asha, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.asha = asha
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        candidate = bracket.update_rungs()

        assert point_hash in bracket.rungs[1][1]
        assert bracket.rungs[1][1][point_hash] == (0.0, (3, 0.0))
        assert candidate[0] == 9

    def test_update_rungs_return_no_candidate(self, asha, bracket, rung_1):
        """Check if no candidate is returned by update_rungs."""
        bracket.asha = asha

        candidate = bracket.update_rungs()

        assert candidate is None

    def test_repr(self, bracket, rung_0, rung_1, rung_2):
        """Test the string representation of Bracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == 'Bracket([1, 3, 9])'


class TestASHA():
    """Tests for the algo ASHA."""

    def test_register(self, asha, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs = [rung_0, rung_1]
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

    def test_register_bracket_multi_fidelity(self, space, b_config):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 1
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        bracket = asha.brackets[0]

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

        fidelity = 3
        point = [fidelity, value]
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[1][1]
        assert (0.0, point) != bracket.rungs[0][1][point_hash]
        assert (0.0, point) == bracket.rungs[1][1][point_hash]

    def test_register_next_bracket(self, space, b_config):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert sum(len(rung[1]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung[1]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung[1]) for rung in asha.brackets[2].rungs) == 0
        assert point_hash in asha.brackets[1].rungs[0][1]
        assert (0.0, point) == asha.brackets[1].rungs[0][1][point_hash]

        value = 51
        fidelity = 9
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert sum(len(rung[1]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung[1]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung[1]) for rung in asha.brackets[2].rungs) == 1
        assert point_hash in asha.brackets[2].rungs[0][1]
        assert (0.0, point) == asha.brackets[2].rungs[0][1][point_hash]

    def test_register_invalid_fidelity(self, space, b_config):
        """Check that a point cannot registered if fidelity is invalid."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with pytest.raises(ValueError) as ex:
            asha.observe([point], [{'objective': 0.0}])

        assert 'No bracket found for point' in str(ex.value)

    def test_register_corrupted_db(self, caplog, space, b_config):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        point = (fidelity, value)

        asha.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' not in caplog.text

        fidelity = 1
        point = [fidelity, value]

        caplog.clear()
        asha.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' in caplog.text

    def test_get_id(self, space, b_config):
        """Test valid id of points"""
        asha = ASHA(space, num_brackets=3)

        assert asha.get_id(['whatever', 1]) == asha.get_id(['is here', 1])
        assert asha.get_id(['whatever', 1]) != asha.get_id(['is here', 2])

    def test_get_id_multidim(self, b_config):
        """Test valid id for points with dim of shape > 1"""
        space = Space()
        space.register(Fidelity('epoch', 1, 9, 3))
        space.register(Real('lr', 'uniform', 0, 1, shape=2))

        asha = ASHA(space, num_brackets=3)

        assert asha.get_id(['whatever', [1, 1]]) == asha.get_id(['is here', [1, 1]])
        assert asha.get_id(['whatever', [1, 1]]) != asha.get_id(['is here', [2, 2]])

    def test_suggest_new(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that a new point is sampled."""
        asha.brackets = [bracket]
        bracket.asha = asha

        def sample(num=1, seed=None):
            return [('fidelity', 0.5)]

        monkeypatch.setattr(asha.space, 'sample', sample)

        points = asha.suggest()

        assert points == [(1, 0.5)]

    def test_suggest_duplicates(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that sampling collisions are handled."""
        asha.brackets = [bracket]
        bracket.asha = asha

        duplicate_point = ('fidelity', 0.0)
        new_point = ('fidelity', 0.5)

        duplicate_id = hashlib.md5(str([duplicate_point]).encode('utf-8')).hexdigest()
        bracket.rungs[0] = (1, {duplicate_id: (0.0, duplicate_point)})

        asha.trial_info[asha.get_id(duplicate_point)] = bracket

        points = [duplicate_point, new_point]

        def sample(num=1, seed=None):
            return [points.pop(0)]

        monkeypatch.setattr(asha.space, 'sample', sample)

        assert asha.suggest()[0][1] == new_point[1]
        assert len(points) == 0

    def test_suggest_inf_duplicates(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that sampling inf collisions raises runtime error."""
        asha.brackets = [bracket]
        bracket.asha = asha

        zhe_point = ('fidelity', 0.0)
        asha.trial_info[asha.get_id(zhe_point)] = bracket

        def sample(num=1, seed=None):
            return [zhe_point]

        monkeypatch.setattr(asha.space, 'sample', sample)

        with pytest.raises(RuntimeError) as exc:
            asha.suggest()

        assert 'ASHA keeps sampling already existing points.' in str(exc.value)

    def test_suggest_in_finite_cardinality(self):
        """Test that suggest None when search space is empty"""
        space = Space()
        space.register(Integer('yolo1', 'uniform', 0, 6))
        space.register(Fidelity('epoch', 1, 9, 3))

        asha = ASHA(space)
        for i in range(6):
            asha.observe([(1, i)], [{'objective': i}])

        for i in range(2):
            asha.observe([(3, i)], [{'objective': i}])

        assert asha.suggest() is None

    def test_suggest_promote(self, asha, bracket, rung_0):
        """Test that correct point is promoted and returned."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        points = asha.suggest()

        assert points == [(3, 0.0)]

    def test_suggest_opt_out(self, asha, bracket, rung_0, rung_1, rung_2):
        """Test that ASHA opts out when last rung is full."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        points = asha.suggest()

        assert points is None

    def test_seed_rng(self, asha):
        """Test that algo is seeded properly"""
        asha.seed_rng(1)
        a = asha.suggest(1)[0]
        assert not np.allclose(a, asha.suggest(1)[0])

        asha.seed_rng(1)
        assert np.allclose(a, asha.suggest(1)[0])

    def test_set_state(self, asha):
        """Test that state is reset properly"""
        asha.seed_rng(1)
        state = asha.state_dict
        point = asha.suggest(1)[0]
        assert not np.allclose(point, asha.suggest(1)[0])

        asha.set_state(state)
        assert point == asha.suggest(1)[0]
