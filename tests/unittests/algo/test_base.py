#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.base`."""

from orion.algo.space import Integer, Real, Space
from orion.core.utils import backward, format_trials


def test_init(dumbalgo):
    """Check if initialization works for nested algos."""
    nested_algo = {"DumbAlgo": dict(value=6, scoring=5)}
    algo = dumbalgo(8, value=1)
    assert algo.space == 8
    assert algo.value == 1
    assert algo.scoring == 0
    assert algo.judgement is None
    assert algo.suspend is False
    assert algo.done is False


def test_configuration(dumbalgo):
    """Check configuration getter works for nested algos."""
    nested_algo = {"DumbAlgo": dict(value=6, scoring=5)}
    algo = dumbalgo(8, value=1)
    config = algo.configuration
    assert config == {
        "dumbalgo": {
            "seed": None,
            "value": 1,
            "scoring": 0,
            "judgement": None,
            "suspend": False,
            "done": False,
        }
    }


def test_state_dict(dumbalgo):
    """Check whether trials_info is in the state dict"""

    space = Space()
    dim = Integer("yolo2", "uniform", -3, 6)
    space.register(dim)
    dim = Real("yolo3", "alpha", 0.9)
    space.register(dim)

    nested_algo = {"DumbAlgo": dict(value=6, scoring=5)}
    algo = dumbalgo(space, value=1)
    algo.suggest(1)
    assert not algo.state_dict["_trials_info"]
    backward.algo_observe(
        algo, [format_trials.tuple_to_trial((1, 2), space)], [dict(objective=3)]
    )
    assert len(algo.state_dict["_trials_info"]) == 1
    backward.algo_observe(
        algo, [format_trials.tuple_to_trial((1, 2), space)], [dict(objective=3)]
    )
    assert len(algo.state_dict["_trials_info"]) == 1


def test_is_done_cardinality(monkeypatch, dumbalgo):
    """Check whether algorithm will stop with base algorithm cardinality check"""
    monkeypatch.delattr(dumbalgo, "is_done")

    space = Space()
    space.register(Integer("yolo1", "uniform", 1, 4))

    algo = dumbalgo(space)
    algo.suggest(6)
    for i in range(1, 6):
        backward.algo_observe(
            algo, [format_trials.tuple_to_trial((i,), space)], [dict(objective=3)]
        )

    assert len(algo.state_dict["_trials_info"]) == 5
    assert algo.is_done

    space = Space()
    space.register(Real("yolo1", "uniform", 1, 4))

    algo = dumbalgo(space)
    algo.suggest(6)
    for i in range(1, 6):
        backward.algo_observe(
            algo, [format_trials.tuple_to_trial((i,), space)], [dict(objective=3)]
        )

    assert len(algo.state_dict["_trials_info"]) == 5
    assert not algo.is_done


def test_is_done_max_trials(monkeypatch, dumbalgo):
    """Check whether algorithm will stop with base algorithm max_trials check"""
    monkeypatch.delattr(dumbalgo, "is_done")

    space = Space()
    space.register(Real("yolo1", "uniform", 1, 4))

    algo = dumbalgo(space)
    algo.suggest(5)
    for i in range(1, 5):
        backward.algo_observe(
            algo, [format_trials.tuple_to_trial((i,), space)], [dict(objective=3)]
        )

    assert len(algo.state_dict["_trials_info"]) == 4
    assert not algo.is_done

    dumbalgo.max_trials = 4
    assert algo.is_done
