#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.strategies`."""
import logging

import pytest

from orion.core.worker.strategy import (
    MaxParallelStrategy,
    MeanParallelStrategy,
    NoParallelStrategy,
    Strategy,
    StubParallelStrategy,
)
from orion.core.worker.trial import Trial


@pytest.fixture
def observations():
    """10 objective observations"""
    points = [i for i in range(10)]
    results = [{"objective": points[i]} for i in range(10)]

    return points, results


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
        Strategy(strategy).observe([corrupted_trial], [{"objective": 1}])
        lie = Strategy(strategy).lie(corrupted_trial)

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
        Strategy(strategy).observe([incomplete_trial], [{"objective": None}])
        Strategy(strategy).lie(incomplete_trial)

    assert "Trial `{}` has an objective but status is not completed" not in caplog.text


class TestStrategyFactory:
    """Test creating a parallel strategy with the Strategy class"""

    def test_create_noparallel(self):
        """Test creating a NoParallelStrategy class"""
        strategy = Strategy("NoParallelStrategy")
        assert isinstance(strategy, NoParallelStrategy)

    def test_create_meanparallel(self):
        """Test creating a MeanParallelStrategy class"""
        strategy = Strategy("MeanParallelStrategy")
        assert isinstance(strategy, MeanParallelStrategy)


class TestParallelStrategies:
    """Test the different parallel strategy methods"""

    def test_max_parallel_strategy(self, observations, incomplete_trial):
        """Test that MaxParallelStrategy lies using the max"""
        points, results = observations

        strategy = MaxParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)

        max_value = max(result["objective"] for result in results)
        assert lying_result.value == max_value

    def test_mean_parallel_strategy(self, observations, incomplete_trial):
        """Test that MeanParallelStrategy lies using the mean"""
        points, results = observations

        strategy = MeanParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)

        mean_value = sum(result["objective"] for result in results) / float(
            len(results)
        )
        assert lying_result.value == mean_value

    def test_no_parallel_strategy(self, observations, incomplete_trial):
        """Test that NoParallelStrategy lies outputs None"""
        points, results = observations

        strategy = NoParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)
        assert lying_result is None

    def test_stub_parallel_strategy(self, observations, incomplete_trial):
        """Test that NoParallelStrategy lies outputs None"""
        points, results = observations

        strategy = StubParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)
        assert lying_result.value is None
