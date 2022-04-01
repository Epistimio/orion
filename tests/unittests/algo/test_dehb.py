"""Perform integration tests for `orion.algo.dehb`."""

import itertools

import numpy
import pytest

from orion.algo.dehb.dehb import IMPORT_ERROR, UnsupportedConfiguration
from orion.core.utils import backward, format_trials
from orion.testing.algo import BaseAlgoTests

if IMPORT_ERROR:
    pytest.skip("skipping DEHB tests", allow_module_level=True)


class TestDEHB(BaseAlgoTests):
    """Test suite for algorithm DEHB"""

    algo_name = "dehb"
    config = {
        "seed": 1234,
        # Because this is so random
        # Add other arguments for your algorithm to pass test_configuration
        "mutation_factor": 0.65,
        "crossover_prob": 0.45,
        "mutation_strategy": "rand2dir",
        "crossover_strategy": "exp",
        "boundary_fix_type": "clip",
        "min_clip": None,
        "max_clip": None,
        "max_age": 12,
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}

    def test_config_mut_strategy_isnot_valid(self):
        with pytest.raises(UnsupportedConfiguration):
            self.create_algo(config=dict(mutation_strategy="123"))

    def test_config_cross_strategy_isnot_valid(self):
        with pytest.raises(UnsupportedConfiguration):
            self.create_algo(config=dict(crossover_strategy="123"))

    def test_config_fix_mode_isnot_valid(self):
        with pytest.raises(UnsupportedConfiguration):
            self.create_algo(config=dict(boundary_fix_type="123"))

    def test_missing_fidelity(self):
        with pytest.raises(RuntimeError):
            space = self.create_space(dict(x="uniform(0, 1)"))
            self.create_algo(space=space)

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        trials = algo.suggest(3)
        assert len(trials) == 3

    @pytest.mark.xfail
    def test_is_done_cardinality(self):
        """Fails because of https://github.com/Epistimio/orion/issues/598"""
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        i = 0
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            backward.algo_observe(
                algo,
                [format_trials.tuple_to_trial([1, x, y, z], space)],
                [dict(objective=i)],
            )
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    # pylint: disable=arguments-differ
    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num):
        space = self.create_space()

        # pylint: disable=invalid-name
        MAX_TRIALS = 10
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


# These are the total number of suggestions that the algorithm will make
# for each "phase" (including previous ones).
# The maximum number is 32 and then it will be done and stop suggesting mode.
# COUNTS = [8 + 4 * 3, 4 + 2 + 4, 2 + 1, 1]
COUNTS = [8 + 4 * 3, 4 + 2 + 4]
COUNTS = numpy.cumsum(COUNTS)

TestDEHB.set_phases(
    [("random", 0, "space.sample")]
    + [(f"rung{i}", budget - 1, "space.sample") for i, budget in enumerate(COUNTS)]
)