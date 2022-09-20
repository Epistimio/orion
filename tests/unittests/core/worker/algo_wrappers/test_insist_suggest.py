from __future__ import annotations

import logging
from typing import ClassVar

import pytest
from pytest import MonkeyPatch

from orion.algo.random import Random
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.algo_wrappers import InsistSuggest
from orion.core.worker.trial import Trial

from .base import AlgoWrapperTests
from .test_transform import FixedSuggestionAlgo


class TestInsistSuggestWrapper(AlgoWrapperTests):
    """Tests for the AlgoWrapper that makes suggest try repeatedly until a new trial is returned."""

    Wrapper: ClassVar[type[InsistSuggest]] = InsistSuggest

    def test_doesnt_insists_without_wrapper(
        self,
        algo_wrapper: InsistSuggest[FixedSuggestionAlgo],
        monkeypatch: MonkeyPatch,
    ):
        """Test that when the algo can't produce a new trial, and there is no InsistWrapper, the
        SpaceTransform wrapper fails to sample a new trial.
        """
        algo_without_wrapper = algo_wrapper.algorithm
        assert isinstance(algo_without_wrapper, FixedSuggestionAlgo)
        calls: int = 0
        # Make the wrapper insist enough so that it actually
        # gets a trial after asking enough times:
        fixed_suggestion = algo_without_wrapper.space.sample(1)[0]

        def _suggest(num: int) -> list[Trial]:
            nonlocal calls
            calls += 1
            if calls < 5:
                return []
            return [fixed_suggestion]

        monkeypatch.setattr(algo_without_wrapper, "suggest", _suggest)
        trials = algo_without_wrapper.suggest(1)
        assert calls == 1
        assert not trials

    def test_insists_when_algo_doesnt_suggest_new_trials(
        self,
        algo_wrapper: InsistSuggest[FixedSuggestionAlgo],
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
            return [algo_wrapper.unwrap(FixedSuggestionAlgo).fixed_suggestion]

        monkeypatch.setattr(algo_wrapper.unwrapped, "suggest", _suggest)
        trial = algo_wrapper.suggest(1)[0]
        assert calls == 5
        assert trial in algo_wrapper.space

    def test_warns_when_unable_to_sample_new_trial(
        self,
        algo_wrapper: InsistSuggest[FixedSuggestionAlgo],
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
            return [algo_wrapper.unwrap(FixedSuggestionAlgo).fixed_suggestion]

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

    def test_doesnt_insist_when_wrapped_algo_is_done(self):
        """Test that when the wrapped algo reaches the `done` stage, the `InsistSuggest` wrapper
        stops asking for new trials.
        """

        space = SpaceBuilder().build({"x": "uniform(1, 10, discrete=True)"})
        algo = Random(space=space)
        algo_wrapper = InsistSuggest(space=space, algorithm=algo)
        algo.max_trials = 10
        assert algo_wrapper.max_trials == 10
        trials = algo_wrapper.suggest(100)
        assert len(trials) == 10
        assert algo_wrapper.is_done
        assert algo_wrapper.n_suggested == 10
