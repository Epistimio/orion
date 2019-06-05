#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.asha`."""

import hashlib
import numpy as np
import pytest

from orion.algo.asha import ASHA, _Bracket
from orion.algo.space import Real, Fidelity, Space


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
    """Return a `_Bracket` instance configured with `b_config`."""
    return _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 1, 9)
    return (1, {hashlib.md5(str([point]).encode('utf-8')).hexdigest():
            (point, [point, 1]) for point in points})


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    return (3, {hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value for value in
            map(lambda v: (v[0], [v[0], 3]), sorted(rung_0[1].values()))})


class TestBracket():
    """Tests for the `_Bracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0][0] == 1
        assert bracket.rungs[1][0] == 3
        assert bracket.rungs[2][0] == 9

    def test_negative_minimum_resources(self, b_config):
        """Test to see if `_Bracket` handles negative minimum resources."""
        b_config['r'] = -1

        with pytest.raises(AttributeError) as ex:
            _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'positive' in str(ex)

    def test_min_resources_greater_than_max(self, b_config):
        """Test to see if `_Bracket` handles minimum resources too high."""
        b_config['r'] = 10

        with pytest.raises(AttributeError) as ex:
            _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'smaller' in str(ex)

    def test_register(self, asha, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.asha = asha
        point = [0.0, 1]
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0][1]
        assert (0.0, point) == bracket.rungs[0][1][point_hash]

    def test_bad_register(self, asha, bracket):
        """Check that a non-valid point is not registered."""
        bracket.asha = asha

        with pytest.raises(AttributeError) as ex:
            bracket.register([0.0, 3], 0.0)

        assert 'Point' in str(ex)

    def test_candidate_promotion(self, asha, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        objective, point = bracket.get_candidate(0)

        assert objective == 0.0
        assert point == [0.0, 1]

    def test_promotion_with_rung_1_hit(self, asha, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = [0.0, 1]
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1][1][point_hash] = (0.0, point)

        objective, point = bracket.get_candidate(0)

        assert objective == 0.125
        assert point == [0.125, 1]

    def test_no_promotion_when_rung_full(self, asha, bracket, rung_0, rung_1):
        """Test that get_candidate returns `None` if rung 1 is full."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        objective, point = bracket.get_candidate(0)

        assert objective is None
        assert point is None
<<<<<<< HEAD

    def test_no_promotion_if_not_enough_points(self, asha, bracket):
        """Test the get_candidate return None if there is not enough points ready."""
        bracket.asha = asha
        bracket.rungs[0] = (1, {hashlib.md5(str([0.0]).encode('utf-8')).hexdigest():
                                (0.0, [0.0, 1])})

        objective, point = bracket.get_candidate(0)

        assert objective is None
        assert point is None

    def test_is_done(self, bracket, rung_0):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2] = (9, {'1': [0.0, 1]})

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, asha, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.asha = asha
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        candidate = bracket.update_rungs()

        assert point_hash in bracket.rungs[2][1]
        assert bracket.rungs[1][1][point_hash] == (0.0, [0.0, 9])
        assert candidate[1] == 9

    def test_update_rungs_return_no_candidate(self, asha, bracket, rung_1):
        """Check if no candidate is returned by update_rungs."""
        bracket.asha = asha

        candidate = bracket.update_rungs()

        assert candidate is None
=======
>>>>>>> b0eb25f... Add test for `get_candidate`
