# pylint: disable=arguments-differ
"""Perform integration tests for `orion.algo.pb2`."""
from __future__ import annotations

from typing import ClassVar

import pytest

from orion.algo.pbt.pb2_utils import HAS_PB2
from orion.testing.algo import BaseAlgoTests, TestPhase

pytest.skip("skipping PBT tests for v0.2.4", allow_module_level=True)


if not HAS_PB2:
    pytest.skip("PB2 deps not installed", allow_module_level=True)

population_size = 10
generations = 5


@pytest.fixture
def no_shutil_copytree(monkeypatch):
    """Pytest fixture copied from orion"""
    monkeypatch.setattr("shutil.copytree", lambda dir_a, dir_b: None)
    yield


# Test suite for algorithms. You may reimplement some of the tests to adapt them to your algorithm
# Full documentation is available at https://orion.readthedocs.io/en/stable/code/testing/algo.html
# Look for algorithms tests in https://github.com/Epistimio/orion/blob/master/tests/unittests/algo
# for examples of customized tests.
@pytest.mark.usefixtures("no_shutil_copytree")
class TestPB2(BaseAlgoTests):
    """Test suite for algorithm PB2"""

    algo_name = "pb2"
    max_trials = population_size * generations
    config = {
        "seed": 123456,
        "population_size": population_size,
        "generations": generations,
        "exploit": {
            "of_type": "PipelineExploit",
            "exploit_configs": [
                {
                    "of_type": "BacktrackExploit",
                    "min_forking_population": population_size / 2,
                    "candidate_pool_ratio": 0.0,
                    "truncation_quantile": 1.0,
                },
                {
                    "of_type": "TruncateExploit",
                    "min_forking_population": population_size / 2,
                    "candidate_pool_ratio": 0.3,
                    "truncation_quantile": 0.9,
                },
            ],
        },
        "fork_timeout": 5,
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=1)"}

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("generation_2", 2 * population_size, "_generate_offspring"),
        TestPhase("generation_3", 3 * population_size, "_generate_offspring"),
    ]

    @pytest.mark.skip(
        reason="There are no good reasons to use PBT if search space is so small"
    )
    def test_is_done_cardinality(self):
        """Ignored."""

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num):
        """Copied from TestGenericPBT"""
        space = self.create_space()

        local_max_trials = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = local_max_trials

        objective = 0
        while not algo.is_done:
            trials = algo.suggest(num)
            assert trials is not None
            if trials:
                self.observe_trials(trials, algo, objective)
                objective += len(trials)

        # BPT should ignore max trials.
        assert algo.n_observed > local_max_trials
        # BPT should stop when all trials of last generation are completed.
        assert algo.n_observed == population_size * (generations + 1)
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        """Ignored."""
