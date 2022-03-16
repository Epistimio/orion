#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.algo.parallel_strategies`."""
import logging

import pytest
import copy
from orion.core.worker.trial import Trial
from orion.algo.parallel_strategy import (
    MaxParallelStrategy,
    MeanParallelStrategy,
    NoParallelStrategy,
    StubParallelStrategy,
    strategy_factory,
)
from orion.core.utils import backward

strategies = [
    "MaxParallelStrategy",
    "MeanParallelStrategy",
    "NoParallelStrategy",
    "StubParallelStrategy",
]


class BaseParallelStrategyTests:
    """Generic Test-suite for parallel strategies.

    This test-suite follow the same logic than  BaseAlgoTests, but applied for ParallelStrategy
    classes.
    """

    parallel_strategy_name = None
    config = {}
    expected_value = None
    default_value = None

    def create_strategy(self, config=None, **kwargs):
        """Create the parallel strategy based on config.

        Parameters
        ----------
        config: dict, optional
            The configuration for the parallel strategy. ``self.config`` will be used
            if ``config`` is ``None``.
        kwargs: dict
            Values to override strategy configuration.
        """
        config = copy.deepcopy(config or self.config)
        config.update(kwargs)
        # NOTE: This previously didn't use `config`!
        # return strategy_factory.create(**self.config)
        return strategy_factory.create(**config)

    def get_trials(self):
        """10 objective observations"""
        trials = []
        for i in range(10):
            trials.append(
                Trial(
                    params=[{"name": "x", "type": "real", "value": i}],
                    results=[{"name": "objective", "type": "objective", "value": i}],
                    status="completed",
                )
            )

        return trials

    def get_noncompleted_trial(self, status="reserved"):
        """Return a single trial without results"""
        return Trial(
            params=[{"name": "a", "type": "integer", "value": 6}], status=status
        )

    def get_corrupted_trial(self):
        """Return a corrupted trial with results but status reserved"""
        return Trial(
            params=[{"name": "a", "type": "integer", "value": 6}],
            results=[{"name": "objective", "type": "objective", "value": 1}],
            status="reserved",
        )

    def test_configuration(self):
        """Test that configuration property attribute contains all class arguments."""
        strategy = self.create_strategy()
        assert strategy.configuration != self.create_strategy(config={})
        assert strategy.configuration == self.config

    def test_state_dict(self):
        """Verify state is restored properly"""
        strategy = self.create_strategy()

        strategy.observe(self.get_trials())

        new_strategy = self.create_strategy()
        assert strategy.state_dict != new_strategy.state_dict

        new_strategy.set_state(strategy.state_dict)
        assert strategy.state_dict == new_strategy.state_dict

        noncompleted_trial = self.get_noncompleted_trial()

        if strategy.infer(noncompleted_trial) is None:
            assert strategy.infer(noncompleted_trial) == new_strategy.infer(
                noncompleted_trial
            )
        else:
            assert (
                strategy.infer(noncompleted_trial).objective.value
                == new_strategy.infer(noncompleted_trial).objective.value
            )

    def test_infer_no_history(self):
        """Test that strategy can infer even without having seen trials"""
        noncompleted_trial = self.get_noncompleted_trial()
        trial = self.create_strategy().infer(noncompleted_trial)
        if self.expected_value is None:
            assert trial is None
        elif self.default_value is None:
            assert trial.objective.value == self.expected_value
        else:
            assert trial.objective.value == self.default_value

    def test_handle_corrupted_trials(self, caplog):
        """Test that strategy can handle trials that has objective but status is not
        properly set to completed."""
        corrupted_trial = self.get_corrupted_trial()
        with caplog.at_level(logging.WARNING, logger="orion.algo.parallel_strategy"):
            trial = self.create_strategy().infer(corrupted_trial)

        match = (
            f"Trial `{corrupted_trial.id}` has an objective but status is not completed"
        )
        assert match in caplog.text

        assert trial is not None
        assert trial.objective.value == corrupted_trial.objective.value

    def test_handle_noncompleted_trials(self, caplog):
        with caplog.at_level(logging.WARNING, logger="orion.algo.parallel_strategy"):
            self.create_strategy().infer(self.get_noncompleted_trial())

        assert (
            "Trial `{}` has an objective but status is not completed" not in caplog.text
        )

    def test_strategy_value(self):
        """Test that ParallelStrategy returns the expected value"""
        strategy = self.create_strategy()
        strategy.observe(self.get_trials())
        trial = strategy.infer(self.get_noncompleted_trial())

        if self.expected_value is None:
            assert trial is None
        else:
            assert trial.objective.value == self.expected_value


class TestNoParallelStrategy(BaseParallelStrategyTests):
    config = {"of_type": "noparallelstrategy"}
    expected_value = None


class TestMaxParallelStrategy(BaseParallelStrategyTests):
    config = {"of_type": "maxparallelstrategy", "default_result": 1000}
    expected_value = 9
    default_value = 1000


class TestMeanParallelStrategy(BaseParallelStrategyTests):
    config = {"of_type": "meanparallelstrategy", "default_result": 1000}
    expected_value = 4.5
    default_value = 1000


class TestStubParallelStrategy(BaseParallelStrategyTests):
    config = {"of_type": "stubparallelstrategy", "stub_value": 3}
    expected_value = 3


class TestStatusBasedParallelStrategy(BaseParallelStrategyTests):
    config = {
        "of_type": "statusbasedparallelstrategy",
        "strategy_configs": {
            "broken": {"of_type": "maxparallelstrategy", "default_result": 1000},
            "suspended": {"of_type": "maxparallelstrategy", "default_result": 100},
        },
        "default_strategy": {"of_type": "meanparallelstrategy", "default_result": 50},
    }
    expected_value = 4.5
    default_value = 50

    def test_routing(self):
        """Test that trials are assigned to proper strategy"""
        strategy = self.create_strategy()
        for status, expected_value in [
            ("broken", 1000),
            ("suspended", 100),
            ("reserved", 50),
        ]:
            assert (
                strategy.infer(
                    self.get_noncompleted_trial(status=status)
                ).objective.value
                == expected_value
            )

        strategy.observe(self.get_trials())
        for status, expected_value in [
            ("broken", 9),
            ("suspended", 9),
            ("reserved", 4.5),
        ]:
            assert (
                strategy.infer(
                    self.get_noncompleted_trial(status=status)
                ).objective.value
                == expected_value
            )
