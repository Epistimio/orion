"""Perform integration tests for `orion.algo.mofa`."""
from __future__ import annotations

from typing import ClassVar

import pytest
import scipy
from packaging import version

from orion.algo.mofa.mofa import MOFA
from orion.algo.space import Space
from orion.testing.algo import BaseAlgoTests, TestPhase


def test_scipy_version():
    try:
        algo = MOFA(Space())
    except RuntimeError:
        assert version.parse(scipy.__version__) < version.parse("1.8")


@pytest.mark.skipif(
    version.parse(scipy.__version__) < version.parse("1.8"),
    reason="requires scipy v1.8 or higher",
)
class TestMOFA(BaseAlgoTests):
    """Test suite for algorithm MOFA"""

    algo_name = "mofa"

    config = {
        "seed": 1234,  # Because this is so random
        "index": 1,
        "n_levels": 5,
        "strength": 2,
        "threshold": 0.1,
    }

    n_trials = config["index"] * config["n_levels"] ** config["strength"]

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("1st-run", 0, "space.sample"),
        TestPhase("2nd-run", n_trials, "space.sample"),
        TestPhase("3rd-run", n_trials * 2, "space.sample"),
    ]

    @classmethod
    def get_num(cls, num):
        return 1

    def test_cfg_index_invalid(self):
        """Tests an invalid index configuration value to MOFA"""
        with pytest.raises(ValueError):
            self.create_algo(config={"index": 0})

    def test_cfg_nlevels_invalid(self):
        """Tests a number of levels which is not prime"""
        with pytest.raises(ValueError):
            self.create_algo(config={"n_levels": 4})

    def test_cfg_strength_lt1_invalid(self):
        """Tests an invalid strength value"""
        with pytest.raises(ValueError):
            self.create_algo(config={"strength": 0})

    def test_cfg_strength_gt2_invalid(self):
        """Tests an invalid strength value"""
        with pytest.raises(ValueError):
            self.create_algo(config={"strength": 3})

    def test_cfg_threshold_le0_invalid(self):
        """Tests an invalid threshold of 0"""
        with pytest.raises(ValueError):
            self.create_algo(config={"threshold": 0})

    def test_cfg_threshold_ge1_invalid(self):
        """Tests and invalid threshold of 1"""
        with pytest.raises(ValueError):
            self.create_algo(config={"threshold": 1})

    def test_cat_data(self):
        if self._current_phase.name == "3rd-run":
            pytest.xfail("MOFA does not explore well categorical dimensions")
        super().test_cat_data()

    @pytest.mark.skip(
        reason="MOFA converges too fast and does not observe the whole space"
    )
    def test_is_done_cardinality(self):
        """Test that algorithm will stop when cardinality is reached"""
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space, index=320, n_levels=2, strength=2)
        trials = []
        for i in range(space.cardinality):
            suggested_trials = algo.suggest(1)
            if not suggested_trials:
                continue
            trial = suggested_trials[0]
            trial.results.append(
                trial.Result(name="objective", value=i, type="objective")
            )
            trial.status = "completed"
            algo.observe([trial])
            trials.append(trial)
            if algo.is_done:
                break

        assert algo.is_done
        assert len(trials) == space.cardinality
