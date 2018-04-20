#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.base`."""

from orion.algo.base import BaseAlgorithm


def test_init(dumbalgo):
    """Check if initialization works for nested algos."""
    nested_algo = {'DumbAlgo': dict(
        value=6,
        scoring=5
        )}
    algo = dumbalgo(8, value=1, subone=nested_algo)
    assert algo.space == 8
    assert algo.value == 1
    assert algo.scoring == 0
    assert algo.judgement is None
    assert algo.suspend is False
    assert algo.done is False
    assert isinstance(algo.subone, BaseAlgorithm)
    assert algo.subone.space == 8
    assert algo.subone.value == 6
    assert algo.subone.scoring == 5


def test_configuration(dumbalgo):
    """Check configuration getter works for nested algos."""
    nested_algo = {'DumbAlgo': dict(
        value=6,
        scoring=5
        )}
    algo = dumbalgo(8, value=1, subone=nested_algo)
    config = algo.configuration
    assert config == {
        'dumbalgo': {
            'value': 1,
            'scoring': 0,
            'judgement': None,
            'suspend': False,
            'done': False,
            'subone': {
                'dumbalgo': {
                    'value': 6,
                    'scoring': 5,
                    'judgement': None,
                    'suspend': False,
                    'done': False,
                    }
                }
            }
        }


def test_space_setter(dumbalgo):
    """Check whether space setter works for nested algos."""
    nested_algo = {'DumbAlgo': dict(
        value=9,
        )}
    nested_algo2 = {'DumbAlgo': dict(
        judgement=10,
        )}
    algo = dumbalgo(8, value=1, naedw=nested_algo, naekei=nested_algo2)
    algo.space = 'etsh'
    assert algo.space == 'etsh'
    assert algo.naedw.space == 'etsh'
    assert algo.naedw.value == 9
    assert algo.naekei.space == 'etsh'
    assert algo.naekei.judgement == 10
