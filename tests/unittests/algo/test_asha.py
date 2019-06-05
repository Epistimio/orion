#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.asha`."""

import pytest

from orion.algo.asha import _Bracket


@pytest.fixture
def b_config():
    return {'n': 9, 'r': 1, 'R': 9, 'eta': 3}


class TestBracket():
    """Tests for the `_Bracket` class."""

    def test_rungs_creation(self, b_config):
        """Test the creation of rungs for bracket 0."""
        bracket = _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert len(bracket.rungs) == 3
        assert bracket.rungs[0][0] == 9
        assert bracket.rungs[1][0] == 3
        assert bracket.rungs[2][0] == 1

    def test_negative_minimum_resources(self, b_config):
        """Test to see if `_Bracket` handles negative minimum resources."""
        b_config['r'] = -1

        with pytest.raises(AttributeError) as ex:
            _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'positive' in str(ex)

    def test_min_resources_greater_than_max(self, b_config):
        b_config['r'] = 10

        with pytest.raises(AttributeError) as ex:
            _Bracket(None, b_config['r'], b_config['R'], b_config['eta'], 0)

        assert 'smaller' in str(ex)
