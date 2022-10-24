#!/usr/bin/env python
"""Collection of tests for :mod:`orion.algo.parallel_strategies`."""


from orion.testing.algo import BaseParallelStrategyTests

strategies = [
    "MaxParallelStrategy",
    "MeanParallelStrategy",
    "NoParallelStrategy",
    "StubParallelStrategy",
]


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
