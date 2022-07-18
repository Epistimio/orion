from __future__ import annotations

import typing
from typing import Any

import pytest
from pytest import MonkeyPatch
from test_transform import FixedSuggestionAlgo

from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.algo_wrappers import InsistSuggest, SpaceTransform
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.trial import Trial

if typing.TYPE_CHECKING:
    from tests.conftest import DumbAlgo


@pytest.fixture()
def palgo(
    dumbalgo: type[DumbAlgo], space: Space, fixed_suggestion_value: Any
) -> InsistSuggest[SpaceTransform[DumbAlgo]]:
    """Set up a SpaceTransformAlgoWrapper with dumb configuration."""
    return create_algo(algo_type=dumbalgo, space=space, value=fixed_suggestion_value)


import logging

from orion.core.utils.format_trials import dict_to_trial


@pytest.fixture()
def algo_wrapper():
    """Fixture that creates the setup for the registration tests below."""
    space = SpaceBuilder().build({"x": "uniform(1, 100, discrete=True)"})
    # NOTE: important, the transformed space will be the same as the original space in this case,
    # so the fixed suggestion will fit.
    return create_algo(
        algo_type=FixedSuggestionAlgo,
        space=space,
        fixed_suggestion=dict_to_trial({"x": 1}, space=space),
    )


class TestInsistSuggestWrapper:
    """Tests for the AlgoWrapper that makes suggest try repeatedly until a new trial is returned."""

    def test_doesnt_insists_without_wrapper(
        self,
        algo_wrapper: InsistSuggest[SpaceTransform[FixedSuggestionAlgo]],
        monkeypatch: MonkeyPatch,
    ):
        """Test that when the algo can't produce a new trial, and there is no InsistWrapper, the
        SpaceTransform wrapper fails to sample a new trial.
        """
        algo_without_wrapper: SpaceTransform[
            FixedSuggestionAlgo
        ] = algo_wrapper.algorithm
        calls: int = 0
        # Make the wrapper insist enough so that it actually
        # gets a trial after asking enough times:
        fixed_suggestion = algo_without_wrapper.unwrapped.space.sample(1)[0]

        def _suggest(num: int) -> list[Trial]:
            nonlocal calls
            calls += 1
            if calls < 5:
                return []
            return [fixed_suggestion]

        monkeypatch.setattr(algo_without_wrapper.algorithm, "suggest", _suggest)
        trials = algo_without_wrapper.suggest(1)
        assert calls == 1
        assert not trials

    def test_insists_when_algo_doesnt_suggest_new_trials(
        self,
        algo_wrapper: InsistSuggest[SpaceTransform[FixedSuggestionAlgo]],
        monkeypatch: MonkeyPatch,
    ):
        """Test that when the algo can't produce a new trial, the wrapper insists and asks again."""
        calls: int = 0
        algo_wrapper.max_suggest_attempts = 10

        # Make the wrapper insist enough so that it actually
        # gets a trial after asking enough times:

        def _suggest(num: int) -> list[Trial]:
            nonlocal calls
            calls += 1
            if calls < 5:
                return []
            return [algo_wrapper.unwrapped.fixed_suggestion]

        monkeypatch.setattr(algo_wrapper.unwrapped, "suggest", _suggest)
        trial = algo_wrapper.suggest(1)[0]
        assert calls == 5
        assert trial in algo_wrapper.space

    def test_warns_when_unable_to_sample_new_trial(
        self,
        algo_wrapper: InsistSuggest[SpaceTransform[FixedSuggestionAlgo]],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: MonkeyPatch,
    ):
        """Test that when the algo can't produce a new trial even after the max number of attempts,
        a warning is logged and an empty list is returned.
        """

        calls: int = 0

        def _suggest(num: int) -> list[Trial]:
            nonlocal calls
            calls += 1
            if calls < 5:
                return []
            return [algo_wrapper.unwrapped.fixed_suggestion]

        monkeypatch.setattr(algo_wrapper.algorithm, "suggest", _suggest)

        algo_wrapper.max_suggest_attempts = 3

        with caplog.at_level(logging.WARNING):
            out = algo_wrapper.suggest(1)
            assert calls == 3
            assert out == []
            assert len(caplog.record_tuples) == 1
            log_record = caplog.record_tuples[0]
            assert log_record[1] == logging.WARNING and log_record[2].startswith(
                "Unable to sample a new trial"
            )
