#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.worker.primary_algo`."""
from __future__ import annotations

import typing
from typing import Any, TypeVar

import pytest

from orion.algo.base import BaseAlgorithm, algo_factory
from orion.algo.space import Space
from orion.core.utils import backward, format_trials
from orion.core.worker.primary_algo import SpaceTransformAlgoWrapper
from orion.core.worker.transformer import build_required_space
from orion.core.worker.trial import Trial
from orion.testing.algo import create_algo

if typing.TYPE_CHECKING:
    from tests.conftest import DumbAlgo


@pytest.fixture()
def palgo(
    dumbalgo: type[DumbAlgo], space: Space, fixed_suggestion_value: Any
) -> SpaceTransformAlgoWrapper[DumbAlgo]:
    """Set up a SpaceTransformAlgoWrapper with dumb configuration."""
    return create_algo(dumbalgo, space=space, value=fixed_suggestion_value)


class TestSpaceTransformAlgoWrapperWraps:
    """Test if SpaceTransformAlgoWrapper is actually wrapping the configured algorithm.

    Does not test for transformations.
    """

    def test_verify_trial(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], space: Space
    ):
        trial = format_trials.tuple_to_trial((["asdfa", 2], 0, 3.5), space)
        palgo._verify_trial(trial)

        with pytest.raises(ValueError, match="not contained in space:"):
            invalid_trial = format_trials.tuple_to_trial((("asdfa", 2), 10, 3.5), space)
            palgo._verify_trial(invalid_trial)

        # transform space
        tspace = build_required_space(
            space, type_requirement="real", shape_requirement="flattened"
        )
        # transform point
        ttrial = tspace.transform(trial)

        ttrial in tspace

        # Transformed point is not in original space
        with pytest.raises(ValueError, match="not contained in space:"):
            palgo._verify_trial(ttrial)

        # Transformed point is in transformed space
        palgo._verify_trial(ttrial, space=tspace)

    def test_init_and_configuration(
        self,
        dumbalgo: type[DumbAlgo],
        palgo: SpaceTransformAlgoWrapper[DumbAlgo],
        fixed_suggestion_value: Trial,
    ):
        """Check if initialization works."""
        assert isinstance(palgo.algorithm, dumbalgo)
        assert palgo.configuration == {
            "dumbalgo": {
                "seed": None,
                "value": fixed_suggestion_value,
                "scoring": 0,
                "judgement": None,
                "suspend": False,
                "done": False,
            }
        }

    def test_space_can_only_retrieved(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], space: Space
    ):
        """Set space is forbidden, getter works as supposed."""
        assert palgo.space == space
        with pytest.raises(AttributeError):
            palgo.space = 5

    def test_suggest(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], fixed_suggestion: Trial
    ):
        """Suggest wraps suggested."""
        palgo.algorithm.pool_size = 10
        trials = palgo.suggest(1)
        assert trials is not None
        assert trials[0].params == fixed_suggestion.params
        ptrials = palgo.suggest(4)
        # NOTE: Now, if an algorithm has already suggested the same trial, we don't return a
        # duplicate.
        assert not ptrials
        palgo.algorithm.possible_values = [fixed_suggestion]
        del fixed_suggestion._params[-1]
        with pytest.raises(ValueError, match="not contained in space"):
            palgo.suggest(1)

    def test_observe(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], fixed_suggestion: Trial
    ):
        """Observe wraps observations."""
        backward.algo_observe(palgo, [fixed_suggestion], [dict(objective=5)])
        palgo.observe([fixed_suggestion])
        assert palgo.algorithm._trials[0].params == fixed_suggestion.params

    def test_isdone(self, palgo: SpaceTransformAlgoWrapper[DumbAlgo]):
        """Wrap isdone."""
        palgo.algorithm.done = 10
        assert palgo.is_done == 10
        assert palgo.algorithm._times_called_is_done == 1

    def test_shouldsuspend(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], fixed_suggestion: Trial
    ):
        """Wrap should_suspend."""
        palgo.algorithm.suspend = 55
        assert palgo.should_suspend(fixed_suggestion) == 55
        assert palgo.algorithm._times_called_suspend == 1

    def test_score(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], fixed_suggestion: Trial
    ):
        """Wrap score."""
        palgo.algorithm.scoring = 60
        assert palgo.score(fixed_suggestion) == 60
        assert palgo.algorithm._score_trial.params == fixed_suggestion.params

    def test_judge(
        self, palgo: SpaceTransformAlgoWrapper[DumbAlgo], fixed_suggestion: Trial
    ):
        """Wrap judge."""
        palgo.algorithm.judgement = "naedw"
        assert palgo.judge(fixed_suggestion, 8) == "naedw"
        assert palgo.algorithm._judge_trial.params == fixed_suggestion.params
        assert palgo.algorithm._measurements == 8
        with pytest.raises(ValueError, match="not contained in space"):
            del fixed_suggestion._params[-1]
            palgo.judge(fixed_suggestion, 8)
