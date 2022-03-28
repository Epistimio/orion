"""Perform integration tests for `orion.algo.mofa`."""
import pytest

from orion.testing.algo import BaseAlgoTests


@pytest.fixture(autouse=True)
def _config(request):
    """Fixture to xfail test_cat_data for 2nd and 3rd runs."""
    if "num" not in request.fixturenames:
        yield
        return

    test_name, _ = request.node.name.split("[")
    if test_name == "test_cat_data" and request.getfixturevalue("num") > 0:
        pytest.xfail("MOFA does not explore well categorical dimensions")

    yield


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

    def get_num(self, num):
        return 1

    def test_config_index_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"index": 0})

    def test_config_nlevels_nonprime_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"n_levels": 4})

    def test_config_strength_lt1_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"strength": 0})

    def test_config_strength_gt2_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"strength": 3})

    def test_config_threshold_le0_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"threshold": 0})

    def test_config_threshold_ge1_isnot_valid(self):
        with pytest.raises(ValueError):
            self.create_algo(config={"threshold": 1})

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


n_trials = (
    TestMOFA.config["index"]
    * TestMOFA.config["n_levels"] ** TestMOFA.config["strength"]
)

TestMOFA.set_phases(
    [
        ("1st-run", 0, "space.sample"),
        ("2nd-run", n_trials, "space.sample"),
        ("3rd-run", n_trials * 2, "space.sample"),
    ]
)
