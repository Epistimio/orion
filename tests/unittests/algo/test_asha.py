#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.asha`."""

import hashlib

import numpy as np
import pytest

from orion.algo.asha import ASHA, Bracket
from orion.algo.space import Fidelity, Real, Space


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Fidelity('epoch'))
    return space


@pytest.fixture
def b_config():
    """Return a configuration for a bracket."""
    return {'n': 9, 'r': 1, 'R': 9, 'eta': 3}


@pytest.fixture
def asha(b_config, space):
    """Return an instance of ASHA."""
    return ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                reduction_factor=b_config['eta'])


@pytest.fixture
def bracket(b_config):
    """Return a `Bracket` instance configured with `b_config`."""
    return Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 1, 9)
    return (1, {hashlib.md5(str([point]).encode('utf-8')).hexdigest():
            (point, (point, 1)) for point in points})


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    return (3, {hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value for value in
            map(lambda v: (v[0], (v[0], 3)), sorted(rung_0[1].values()))})


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    return (9, {hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value for value in
            map(lambda v: (v[0], (v[0], 9)), sorted(rung_1[1].values()))})


class TestBracket():
    """Tests for the `Bracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0][0] == 1
        assert bracket.rungs[1][0] == 3
        assert bracket.rungs[2][0] == 9

    def test_negative_minimum_resources(self, b_config):
        """Test to see if `Bracket` handles negative minimum resources."""
        b_config['r'] = -1

        with pytest.raises(AttributeError) as ex:
            Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'positive' in str(ex.value)

    def test_min_resources_greater_than_max(self, b_config):
        """Test to see if `Bracket` handles minimum resources too high."""
        b_config['r'] = 10

        with pytest.raises(AttributeError) as ex:
            Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'smaller' in str(ex.value)

    def test_register(self, asha, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.asha = asha
        point = (0.0, 1)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

    def test_bad_register(self, asha, bracket):
        """Check that a non-valid point is not registered."""
        bracket.asha = asha

        with pytest.raises(IndexError) as ex:
            bracket.register((0.0, 55), 0.0)

        assert 'Bad fidelity level 55' in str(ex.value)

    def test_candidate_promotion(self, asha, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        point = bracket.get_candidate(0)

        assert point == (0.0, 1)

    def test_promotion_with_rung_1_hit(self, asha, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = (0.0, 1)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1][1][point_hash] = (0.0, point)

        point = bracket.get_candidate(0)

        assert point == (0.125, 1)

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
                                (0.0, (0.0, 1))})

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
        bracket.rungs[2] = (9, {'1': (0.0, 1)})

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, asha, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.asha = asha
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        candidate = bracket.update_rungs()

        assert point_hash in bracket.rungs[1][1]
        assert bracket.rungs[1][1][point_hash] == (0.0, (0.0, 3))
        assert candidate[1] == 9

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
        point = (0.0, 1)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

    def test_register_bracket_multi_fidelity(self, space, b_config):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        value = 50
        fidelity = 1
        point = (value, fidelity)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        bracket = asha.brackets[0]

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

        fidelity = 3
        point = [value, fidelity]
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[1][1]
        assert (0.0, point) != bracket.rungs[0][1][point_hash]
        assert (0.0, point) == bracket.rungs[1][1][point_hash]

    def test_register_next_bracket(self, space, b_config):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        value = 50
        fidelity = 3
        point = (value, fidelity)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert sum(len(rung[1]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung[1]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung[1]) for rung in asha.brackets[2].rungs) == 0
        assert point_hash in asha.brackets[1].rungs[0][1]
        assert (0.0, point) == asha.brackets[1].rungs[0][1][point_hash]

        value = 51
        fidelity = 9
        point = (value, fidelity)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        asha.observe([point], [{'objective': 0.0}])

        assert sum(len(rung[1]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung[1]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung[1]) for rung in asha.brackets[2].rungs) == 1
        assert point_hash in asha.brackets[2].rungs[0][1]
        assert (0.0, point) == asha.brackets[2].rungs[0][1][point_hash]

    def test_register_invalid_fidelity(self, space, b_config):
        """Check that a point cannot registered if fidelity is invalid."""
        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        value = 50
        fidelity = 2
        point = (value, fidelity)

        with pytest.raises(ValueError) as ex:
            asha.observe([point], [{'objective': 0.0}])

        assert 'No bracket found for point' in str(ex.value)

    def test_register_corrupted_db(self, caplog, space, b_config):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        value = 50
        fidelity = 3
        point = (value, fidelity)

        asha.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' not in caplog.text

        fidelity = 1
        point = [value, fidelity]

        caplog.clear()
        asha.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' in caplog.text

    def test_get_id(self, space, b_config):
        """Test valid id of points"""
        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        assert asha.get_id([1, 'whatever']) == asha.get_id([1, 'is here'])
        assert asha.get_id([1, 'whatever']) != asha.get_id([2, 'is here'])

    def test_get_id_multidim(self, b_config):
        """Test valid id for points with dim of shape > 1"""
        space = Space()
        space.register(Fidelity('epoch'))
        space.register(Real('lr', 'uniform', 0, 1, shape=2))

        asha = ASHA(space, max_resources=b_config['R'], grace_period=b_config['r'],
                    reduction_factor=b_config['eta'], num_brackets=3)

        assert asha.get_id(['whatever', [1, 1]]) == asha.get_id(['is here', [1, 1]])
        assert asha.get_id(['whatever', [1, 1]]) != asha.get_id(['is here', [2, 2]])

    def test_suggest_new(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that a new point is sampled."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        def sample(num=1, seed=None):
            return [(0.5, 'fidelity')]

        monkeypatch.setattr(asha.space, 'sample', sample)

        points = asha.suggest()

        assert points == [(0.5, 1)]

    def test_suggest_duplicates(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that sampling collisions are handled."""
        asha.brackets = [bracket]
        bracket.asha = asha

        # Fill rungs to force sampling
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        duplicate_point = (0.0, 'fidelity')
        new_point = (0.5, 'fidelity')

        asha.trial_info[asha.get_id(duplicate_point)] = bracket

        points = [duplicate_point, new_point]

        def sample(num=1, seed=None):
            return [points.pop(0)]

        monkeypatch.setattr(asha.space, 'sample', sample)

        assert asha.suggest()[0][0] == new_point[0]
        assert len(points) == 0

    def test_suggest_inf_duplicates(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that sampling inf collisions raises runtime error."""
        asha.brackets = [bracket]
        bracket.asha = asha

        # Fill rungs to force sampling
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        zhe_point = (0.0, 'fidelity')
        asha.trial_info[asha.get_id(zhe_point)] = bracket

        def sample(num=1, seed=None):
            return [zhe_point]

        monkeypatch.setattr(asha.space, 'sample', sample)

        with pytest.raises(RuntimeError) as exc:
            asha.suggest()

        assert 'ASHA keeps sampling already existing points.' in str(exc.value)

    def test_suggest_promote(self, asha, bracket, rung_0):
        """Test that correct point is promoted and returned."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        points = asha.suggest()

        assert points == [(0.0, 3)]

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
