#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.strategies`."""
import logging

import pytest

from orion.core.utils import backward
from orion.core.worker.strategy import (
    MaxParallelStrategy,
    MeanParallelStrategy,
    NoParallelStrategy,
    StubParallelStrategy,
    strategy_factory,
)
from orion.core.worker.trial import Trial


@pytest.fixture
def trials():
    """10 objective observations"""
    trials = []
    for i in range(10):
        trials.append(
            Trial(
                params=[{"name": "x", "type": "real", "value": i}],
                results=[{"name": "objective", "type": "objective", "value": i}],
            )
        )

    return trials


@pytest.fixture
def incomplete_trial():
    """Return a single trial without results"""
    return Trial(params=[{"name": "a", "type": "integer", "value": 6}])


@pytest.fixture
def corrupted_trial():
    """Return a corrupted trial with results but status reserved"""
    return Trial(
        params=[{"name": "a", "type": "integer", "value": 6}],
        results=[{"name": "objective", "type": "objective", "value": 1}],
        status="reserved",
    )


strategies = [
    "MaxParallelStrategy",
    "MeanParallelStrategy",
    "NoParallelStrategy",
    "StubParallelStrategy",
]


@pytest.mark.parametrize("strategy", strategies)
def test_handle_corrupted_trials(caplog, strategy, corrupted_trial):
    """Verify that corrupted trials are handled properly"""
    with caplog.at_level(logging.WARNING, logger="orion.core.worker.strategy"):
        lie = strategy_factory.create(strategy).lie(corrupted_trial)

    match = "Trial `{}` has an objective but status is not completed".format(
        corrupted_trial.id
    )
    assert match in caplog.text

    assert lie is not None
    assert lie.value == corrupted_trial.objective.value


@pytest.mark.parametrize("strategy", strategies)
def test_handle_uncompleted_trials(caplog, strategy, incomplete_trial):
    """Verify that no warning is logged if trial is valid"""
    with caplog.at_level(logging.WARNING, logger="orion.core.worker.strategy"):
        strategy_factory.create(strategy).lie(incomplete_trial)

    assert "Trial `{}` has an objective but status is not completed" not in caplog.text


class TestStrategyFactory:
    """Test creating a parallel strategy with the Strategy class"""

    def test_create_noparallel(self):
        """Test creating a NoParallelStrategy class"""
        strategy = strategy_factory.create("NoParallelStrategy")
        assert isinstance(strategy, NoParallelStrategy)

    def test_create_meanparallel(self):
        """Test creating a MeanParallelStrategy class"""
        strategy = strategy_factory.create("MeanParallelStrategy")
        assert isinstance(strategy, MeanParallelStrategy)


class TestParallelStrategies:
    """Test the different parallel strategy methods"""

    def test_max_parallel_strategy(self, trials, incomplete_trial):
        """Test that MaxParallelStrategy lies using the max"""
        strategy = MaxParallelStrategy()
        strategy.observe(trials)
        lying_result = strategy.lie(incomplete_trial)

        max_value = max(trial.objective.value for trial in trials)
        assert lying_result.value == max_value

    def test_mean_parallel_strategy(self, trials, incomplete_trial):
        """Test that MeanParallelStrategy lies using the mean"""
        strategy = MeanParallelStrategy()
        strategy.observe(trials)
        lying_result = strategy.lie(incomplete_trial)

        mean_value = sum(trial.objective.value for trial in trials) / float(len(trials))
        assert lying_result.value == mean_value

    def test_no_parallel_strategy(self, trials, incomplete_trial):
        """Test that NoParallelStrategy lies outputs None"""
        strategy = NoParallelStrategy()
        strategy.observe(trials)
        lying_result = strategy.lie(incomplete_trial)
        assert lying_result is None

    def test_stub_parallel_strategy(self, trials, incomplete_trial):
        """Test that NoParallelStrategy lies outputs None"""
        strategy = StubParallelStrategy()
        strategy.observe(trials)
        lying_result = strategy.lie(incomplete_trial)
        assert lying_result.value is None
