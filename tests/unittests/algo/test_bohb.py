"""Perform integration tests for `orion.algo.bohb`."""
import pytest

from orion.algo.bohb import import_optional
from orion.testing.algo import BaseAlgoTests

if import_optional.failed:
    pytest.skip("skipping BOHB tests", allow_module_level=True)

N_INIT = 5


class TestBOHB(BaseAlgoTests):
    """Test suite for algorithm BOHB"""

    algo_name = "bohb"
    config = {
        "seed": 1234,  # Because this is so random
        # Add other arguments for your algorithm to pass test_configuration
        "min_points_in_model": N_INIT,
        "top_n_percent": 10,
        "num_samples": N_INIT * 2,
        "random_fraction": 1 / 4,
        "bandwidth_factor": 4,
        "min_bandwidth": 1e-4,
        "parallel_strategy": {
            "of_type": "StatusBasedParallelStrategy",
            "strategy_configs": {"broken": {"of_type": "MaxParallelStrategy"}},
        },
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}

    def test_missing_fidelity(self):
        with pytest.raises(RuntimeError):
            space = self.create_space(dict(x="uniform(0, 1)"))
            self.create_algo(space=space)

    def test_default_strategy(self):
        algo = self.create_algo(config=dict(parallel_strategy=None))
        assert algo.algorithm.strategy.configuration == {
            "of_type": "statusbasedparallelstrategy",
            "default_strategy": {"of_type": "noparallelstrategy"},
            "strategy_configs": {
                "broken": {
                    "default_result": float("inf"),
                    "of_type": "maxparallelstrategy",
                },
            },
        }

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/598")
    def test_is_done_cardinality(self):
        pass

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num):  # pylint: disable=arguments-differ
        space = self.create_space()

        MAX_TRIALS = 5  # pylint: disable=invalid-name
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        objective = 0
        while not algo.is_done:
            trials = algo.suggest(num)
            assert trials is not None
            if trials:
                self.observe_trials(trials, algo, objective)
                objective += len(trials)

        # Hyperband should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        self.spy_phase(mocker, num, algo, attr)
        trials = algo.suggest(3)
        assert len(trials) == 3


# These are the total number of suggestions that the algorithm will make
# for each "phase" (including previous ones).
# The maximum number is 32 and then it will be done and stop suggesting mode.
COUNTS = [8, 15, 22, 28]

TestBOHB.set_phases(
    [("random", 0, "space.sample")]
    + [(f"rung{i}", budget, "suggest") for i, budget in enumerate(COUNTS)]
)
