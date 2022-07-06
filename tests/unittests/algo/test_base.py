#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.algo.base`."""

import pytest

from orion.algo.base import BaseAlgorithm
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
    algo = dumbalgo(space, value=(1, 1))
    algo.suggest(1)
    assert not algo.state_dict["registry"]["_trials"]
    backward.algo_observe(
        algo, [format_trials.tuple_to_trial((1, 2), space)], [dict(objective=3)]
    )
    assert len(algo.state_dict["registry"]["_trials"]) == 1
    backward.algo_observe(
        algo, [format_trials.tuple_to_trial((1, 2), space)], [dict(objective=3)]
    )
    assert len(algo.state_dict["registry"]["_trials"]) == 1


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

    assert len(algo.state_dict["registry"]["_trials"]) == 5
    assert algo.is_done

    space = Space()
    space.register(Real("yolo1", "uniform", 1, 4))

    algo = dumbalgo(space)
    algo.suggest(6)
    for i in range(1, 6):
        backward.algo_observe(
            algo, [format_trials.tuple_to_trial((i,), space)], [dict(objective=3)]
        )

    assert len(algo.state_dict["registry"]["_trials"]) == 5
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

    assert len(algo.state_dict["registry"]["_trials"]) == 4
    assert not algo.is_done

    dumbalgo.max_trials = 4
    assert algo.is_done


@pytest.mark.parametrize("pass_to_super", [True, False])
def test_arg_names(pass_to_super: bool):
    """Test that the `_arg_names` can be determined programmatically when the args aren't passed to
    `super().__init__(space, **kwargs)`.

    Also checks that the auto-generated configuration dict acts the same way.
    """

    class SomeAlgo(BaseAlgorithm):
        def __init__(self, space, foo: int = 123, bar: str = "heyo"):
            if pass_to_super:
                super().__init__(space, foo=foo, bar=bar)
            else:
                super().__init__(space)
                self.foo = foo
                self.bar = bar
            # Param names should be correct, either way.
            assert self._param_names == ["foo", "bar"]
            # Attributes should be set correctly either way:
            assert self.foo == foo
            assert self.bar == bar

    space = Space(x=Real("yolo1", "uniform", 1, 4))
    algo = SomeAlgo(space, foo=111, bar="barry")
    assert algo.configuration == {
        "somealgo": {
            "bar": "barry",
            "foo": 111,
        }
    }
