#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.worker.primary_algo`."""
from __future__ import annotations

import copy
import typing
from typing import Any, ClassVar

import numpy
import pytest

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils import backward, format_trials
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.transformer import build_required_space
from orion.core.worker.trial import Trial
from orion.testing.dummy_algo import FixedSuggestionAlgo

if typing.TYPE_CHECKING:
    from tests.conftest import DumbAlgo


@pytest.fixture()
def palgo(
    dumbalgo: type[DumbAlgo], space: Space, fixed_suggestion_value: Any
) -> SpaceTransform[DumbAlgo]:
    """Set up a SpaceTransform with dumb configuration."""
    return create_algo(
        algo_type=dumbalgo, space=space, value=fixed_suggestion_value
    ).algorithm


class TestSpaceTransformWraps:
    """Test if the SpaceTransform wrapper is actually wrapping the configured algorithm.

    Does not test for transformations.
    """

    def test_verify_trial(self, palgo: SpaceTransform[DumbAlgo], space: Space):
        trial = format_trials.tuple_to_trial((["asdfa", 2], 0, 3.5), space)
        palgo._verify_trial(trial)

        assert palgo.space is space

        with pytest.raises(ValueError, match="not contained in space:"):
            invalid_trial = format_trials.tuple_to_trial((("asdfa", 2), 10, 3.5), space)
            palgo._verify_trial(invalid_trial)

        # transform space
        tspace = build_required_space(
            space, type_requirement="real", shape_requirement="flattened"
        )

        # transform point
        ttrial = tspace.transform(trial)
        # TODO: https://github.com/Epistimio/orion/issues/804
        assert ttrial in tspace

        # Transformed point is not in original space
        with pytest.raises(ValueError, match="not contained in space:"):
            palgo._verify_trial(ttrial)

        # Transformed point is in transformed space
        palgo._verify_trial(ttrial, space=tspace)

    def test_init_and_configuration(
        self,
        dumbalgo: type[DumbAlgo],
        palgo: SpaceTransform[DumbAlgo],
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
        self, palgo: SpaceTransform[DumbAlgo], space: Space
    ):
        """Set space is forbidden, getter works as supposed."""
        assert palgo.space == space
        with pytest.raises(AttributeError):
            palgo.space = 5

    def test_suggest(self, palgo: SpaceTransform[DumbAlgo], fixed_suggestion: Trial):
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

    def test_observe(self, palgo: SpaceTransform[DumbAlgo], fixed_suggestion: Trial):
        """Observe wraps observations."""
        backward.algo_observe(palgo, [fixed_suggestion], [dict(objective=5)])
        palgo.observe([fixed_suggestion])
        assert palgo.algorithm._trials[0].params == fixed_suggestion.params

    def test_isdone(self, palgo: SpaceTransform[DumbAlgo]):
        """Wrap isdone."""
        palgo.algorithm.done = 10
        assert palgo.is_done == 10
        assert palgo.algorithm._times_called_is_done == 1

    def test_shouldsuspend(
        self, palgo: SpaceTransform[DumbAlgo], fixed_suggestion: Trial
    ):
        """Wrap should_suspend."""
        palgo.algorithm.suspend = 55
        assert palgo.should_suspend(fixed_suggestion) == 55
        assert palgo.algorithm._times_called_suspend == 1

    def test_score(self, palgo: SpaceTransform[DumbAlgo], fixed_suggestion: Trial):
        """Wrap score."""
        palgo.algorithm.scoring = 60
        assert palgo.score(fixed_suggestion) == 60
        assert palgo.algorithm._score_trial.params == fixed_suggestion.params

    def test_judge(self, palgo: SpaceTransform[DumbAlgo], fixed_suggestion: Trial):
        """Wrap judge."""
        palgo.algorithm.judgement = "naedw"
        assert palgo.judge(fixed_suggestion, 8) == "naedw"
        assert palgo.algorithm._judge_trial.params == fixed_suggestion.params
        assert palgo.algorithm._measurements == 8
        with pytest.raises(ValueError, match="not contained in space"):
            del fixed_suggestion._params[-1]
            palgo.judge(fixed_suggestion, 8)


class GenealogistAlgo(BaseAlgorithm):
    """An algo that always returns a child trial branched from the parent."""

    requires_type: ClassVar[str | None] = "real"
    requires_shape: ClassVar[str | None] = "flattened"
    requires_dist: ClassVar[str | None] = "linear"

    def __init__(
        self,
        space: Space,
        base_suggestion: Trial | None = None,
    ):
        super().__init__(space)
        self.base_suggestion = base_suggestion

    def suggest(self, num):
        new_trial = self.space.sample(1)[0]
        if self.base_suggestion is not None:
            self.base_suggestion = self.base_suggestion.branch(params=new_trial.params)
        else:
            self.base_suggestion = new_trial
        self.register(self.base_suggestion)
        return [self.base_suggestion]


@pytest.fixture()
def algo_wrapper():
    """Fixture that creates the setup for the registration tests below."""
    original_space = SpaceBuilder().build({"x": "loguniform(1, 100, discrete=True)"})

    transformed_space = build_required_space(
        original_space=original_space,
        type_requirement=FixedSuggestionAlgo.requires_type,
        shape_requirement=FixedSuggestionAlgo.requires_shape,
        dist_requirement=FixedSuggestionAlgo.requires_dist,
    )

    fixed_original = original_space.sample(1)[0]
    fixed_transformed = transformed_space.transform(fixed_original)

    algo = FixedSuggestionAlgo(
        space=transformed_space, fixed_suggestion=fixed_transformed
    )

    algo_wrapper = SpaceTransform(algorithm=algo, space=original_space)
    assert algo_wrapper.algorithm is algo
    return algo_wrapper


class TestRegistration:
    """Tests for the new `registry` and `registry_mapping` of the transform algo wrapper."""

    @pytest.mark.parametrize("original_status", ["completed", "broken", "pending"])
    def test_suggest_equivalent_to_existing(
        self, algo_wrapper: SpaceTransform[FixedSuggestionAlgo], original_status: str
    ):
        """Test the case where the underlying algo suggests a transformed trial that is equivalent
        to a previously-suggested trial in the original space.

        In this case, the wrapped algo's `observe` method should be called inside the wrapper's
        `suggest`, passing in the suggested transformed trial, but with the results and status
        copied over from the existing `Trial`.
        """

        fixed_original = format_trials.dict_to_trial(
            {"x": 10}, space=algo_wrapper.space
        )
        fixed_transformed = algo_wrapper.transformed_space.transform(fixed_original)
        assert fixed_transformed.params == {"x": numpy.log(10)}
        transformed_space = algo_wrapper.transformed_space

        algo_wrapper.unwrapped.fixed_suggestion = fixed_transformed

        # Get a suggestion from the wrapper.
        suggested_trials = algo_wrapper.suggest(1)
        assert suggested_trials is not None
        assert suggested_trials == [fixed_original]

        assert fixed_original in algo_wrapper.registry
        assert fixed_original in algo_wrapper.registry_mapping
        assert algo_wrapper.registry_mapping[fixed_original] == [fixed_transformed]
        assert algo_wrapper.has_suggested(fixed_original)
        assert not algo_wrapper.has_observed(fixed_original)

        assert algo_wrapper.algorithm.has_suggested(fixed_transformed)
        assert not algo_wrapper.algorithm.has_observed(fixed_transformed)

        # Now, add some results to the first suggestion, and have the wrapper observe it.
        trial_with_results = copy.deepcopy(fixed_original)
        trial_with_results._status = original_status
        if original_status == "completed":
            trial_with_results._results = [
                Trial.Result(name="objective", type="objective", value=1)
            ]

        trial_with_results_transformed = transformed_space.transform(trial_with_results)
        assert trial_with_results_transformed.results == trial_with_results.results

        algo_wrapper.observe([trial_with_results])

        if original_status in ["completed", "broken"]:
            assert algo_wrapper.has_observed(trial_with_results)
            assert algo_wrapper.algorithm.has_observed(trial_with_results_transformed)

        # Pretty obvious, since this was true before observing. Checking again, just to be sure.
        assert algo_wrapper.has_suggested(trial_with_results)
        assert algo_wrapper.algorithm.has_suggested(trial_with_results_transformed)

        # Create a Trial in the transformed space that will map to the same trial as the fixed
        # suggestion in the original space.
        equivalent_transformed = format_trials.dict_to_trial(
            {"x": fixed_transformed.params["x"] + 1e-6},
            space=transformed_space,
        )
        # Make sure that both are distinct, but map to the same trial in the original space.
        assert equivalent_transformed != fixed_transformed
        assert equivalent_transformed.params != fixed_transformed.params
        equivalent_original = transformed_space.reverse(equivalent_transformed)
        assert equivalent_original == fixed_original

        # Check that has_observed/has_suggested is False for the equivalent trial in the algorithm.
        assert not algo_wrapper.algorithm.has_suggested(equivalent_transformed)
        assert not algo_wrapper.algorithm.has_observed(equivalent_transformed)

        # Update the dummy algo so that it now returns the equivalent transformed trial.
        algo_wrapper.unwrapped.fixed_suggestion = equivalent_transformed

        # Get another suggestion from the wrapper. (the wrapped algo will return the equivalent)
        new_suggested_trials = algo_wrapper.suggest(1)
        assert new_suggested_trials is not None
        # Here the algo wrapper should return an empty list, since the suggested trial is a
        # duplicate of one that was already suggested.
        assert new_suggested_trials == []

        # Check that the equivalent trial was added into the algorithm's registry, and that the
        # status and results from `trial_with_results` were also copied over.
        assert equivalent_transformed in algo_wrapper.algorithm.registry
        trial_in_registry = algo_wrapper.algorithm.registry.get_existing(
            equivalent_transformed
        )

        # NOTE: The algo has suggested a duplicate. We make the algo observe the duplicate, with
        # the status (and potentially also results) from the original. This is also the case for
        # broken or pending trials.

        assert trial_in_registry.params == equivalent_transformed.params
        assert trial_in_registry.status == trial_with_results.status
        assert trial_in_registry.results == trial_with_results.results

        # The algo wrapper has suggested (and observed) a trial that is equivalent to this one.
        if original_status in ["completed", "broken"]:
            assert algo_wrapper.has_observed(equivalent_original)
            assert algo_wrapper.algorithm.has_observed(equivalent_transformed)

        assert algo_wrapper.has_suggested(equivalent_original)
        assert algo_wrapper.algorithm.has_suggested(equivalent_transformed)

    def test_suggest_nonexistent_parents(self):
        original_space = SpaceBuilder().build(
            {"x": "loguniform(1, 100, discrete=True)"}
        )
        transformed_space = build_required_space(
            original_space=original_space,
            type_requirement=GenealogistAlgo.requires_type,
            shape_requirement=GenealogistAlgo.requires_shape,
            dist_requirement=GenealogistAlgo.requires_dist,
        )

        base_original = format_trials.dict_to_trial({"x": 10}, space=original_space)
        base_transformed = transformed_space.transform(base_original)
        assert base_transformed.params == {"x": numpy.log(10)}

        algo = GenealogistAlgo(
            space=transformed_space, base_suggestion=base_transformed
        )

        algo_wrapper = SpaceTransform(algorithm=algo, space=original_space)

        with pytest.raises(
            KeyError,
            match=f"Parent trial with id {base_transformed.id} is not registered",
        ):
            _ = algo_wrapper.suggest(1)

    def test_suggest_parents(self):
        original_space = SpaceBuilder().build(
            {"x": "loguniform(1, 100, discrete=True)"}
        )
        transformed_space = build_required_space(
            original_space=original_space,
            type_requirement=GenealogistAlgo.requires_type,
            shape_requirement=GenealogistAlgo.requires_shape,
            dist_requirement=GenealogistAlgo.requires_dist,
        )

        base_original = format_trials.dict_to_trial({"x": 10}, space=original_space)
        base_transformed = transformed_space.transform(base_original)
        assert base_transformed.params == {"x": numpy.log(10)}

        algo = GenealogistAlgo(space=transformed_space, base_suggestion=None)

        algo_wrapper = SpaceTransform(algorithm=algo, space=original_space)
        assert algo_wrapper.algorithm is algo

        # Get a suggestion from the wrapper.
        suggested_trials = algo_wrapper.suggest(1)
        assert suggested_trials[0] is not None

        for i in range(5):
            suggested_trials.extend(algo_wrapper.suggest(1))
            assert suggested_trials[i + 1] is not None
            assert suggested_trials[i + 1].parent == suggested_trials[i].id

    def test_observe_trial_not_suggested(
        self, algo_wrapper: SpaceTransform[FixedSuggestionAlgo]
    ):
        """Test the case where the wrapper observes a trial that hasn't been suggested by the
        wrapped algorithm.
        """
        original_space = algo_wrapper.space
        transformed_space = algo_wrapper.transformed_space
        algo = algo_wrapper.algorithm

        trial = original_space.sample()[0]
        trial_with_results = copy.deepcopy(trial)
        trial_with_results.status = "completed"
        trial_with_results._results = [
            Trial.Result(name="objective", type="objective", value=1)
        ]
        transformed_trial_with_results = transformed_space.transform(trial_with_results)

        algo_wrapper.observe([trial_with_results])

        assert algo_wrapper.has_observed(trial_with_results)
        assert algo_wrapper.algorithm.has_observed(transformed_trial_with_results)

        assert algo_wrapper.has_suggested(trial_with_results)
        assert algo_wrapper.algorithm.has_suggested(transformed_trial_with_results)
