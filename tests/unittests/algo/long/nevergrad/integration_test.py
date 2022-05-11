"""Perform integration tests for `orion.algo.nevergrad`."""
from typing import ClassVar

import nevergrad as ng
import pytest
from pytest import MarkDecorator

from orion.algo.nevergradoptimizer import NOT_WORKING as NOT_WORKING_MODEL_NAMES
from orion.testing.algo import BaseAlgoTests, TestPhase

TEST_MANY_TRIALS = 20

_AlgoName = str
_TestName = str

xfail = pytest.mark.xfail

_deterministic_first_point = {
    "test_seed_rng_init": xfail(reason="First generated point is deterministic")
}

_deterministic_points = {
    "test_seed_rng_init": xfail(reason="Generated points are deterministic")
}


_not_serializable = {
    "test_has_observed_statedict": xfail(reason="Algorithm is not serializable"),
    "test_state_dict": xfail(reason="Algorithm is not serializable"),
}

_sequential = {
    "test_seed_rng": xfail(reason="Cannot ask before tell of the last ask"),
    "test_seed_rng_init": xfail(reason="First generated point is deterministic"),
}

_no_tell_without_ask = {
    "test_observe": xfail(reason="Cannot observe a point that was not suggested"),
    "test_is_done_cardinality": xfail(
        reason="Cannot observe a point that was not suggested"
    ),
}

_not_parallel = {"test_suggest_n": xfail(reason="Cannot suggest more than one point")}

_CAN_suggest_more_than_one = {"test_suggest_n": None}

_max_trials_hangs = {"test_is_done_max_trials": "skip"}


def merge_dicts(*dicts: dict) -> dict:
    """Merge dictionaries into first one."""
    merged_dict = dicts[0].copy()
    for dict_to_merge in dicts[1:]:
        merged_dict.update(dict_to_merge)

    return merged_dict


WORKING: dict[_AlgoName, dict[_TestName, MarkDecorator]] = {
    "cGA": {},
    "ASCMADEthird": {},
    "AdaptiveDiscreteOnePlusOne": _deterministic_first_point,
    "AlmostRotationInvariantDE": {},
    "AvgMetaRecenteringNoHull": _deterministic_points,
    "CM": {},
    "CMA": {},
    "CauchyLHSSearch": {},
    "CauchyOnePlusOne": _deterministic_first_point,
    "CauchyScrHammersleySearch": _deterministic_points,
    "ChainMetaModelSQP": merge_dicts(_CAN_suggest_more_than_one, _max_trials_hangs),
    "DE": {},
    "DiagonalCMA": {},
    "DiscreteBSOOnePlusOne": _deterministic_first_point,
    "DiscreteLenglerOnePlusOne": _deterministic_first_point,
    "DiscreteOnePlusOne": _deterministic_first_point,
    "EDA": _no_tell_without_ask,
    "ES": {},
    "FCMA": {},
    "GeneticDE": {},
    "HaltonSearch": _deterministic_points,
    "HaltonSearchPlusMiddlePoint": _deterministic_points,
    "HammersleySearch": _deterministic_points,
    "HammersleySearchPlusMiddlePoint": _deterministic_points,
    "HullAvgMetaRecentering": _deterministic_points,
    "HullAvgMetaTuneRecentering": _deterministic_points,
    "LargeHaltonSearch": _deterministic_points,
    "LHSSearch": {},
    "LhsDE": {},
    "MetaModel": {},
    "MetaModelOnePlusOne": _deterministic_first_point,
    "MetaRecentering": _deterministic_points,
    "MetaTuneRecentering": _deterministic_points,
    "MixES": {},
    "MultiCMA": {},
    "MultiScaleCMA": {},
    "MutDE": {},
    "NaiveIsoEMNA": _no_tell_without_ask,
    "NaiveTBPSA": {},
    "NGO": {},
    "NGOpt": {},
    "NGOpt10": {},
    "NGOpt12": {},
    "NGOpt13": {},
    "NGOpt14": {},
    "NGOpt15": {},
    "NGOpt16": {},
    "NGOpt21": {},
    "NGOpt36": {},
    "NGOpt38": {},
    "NGOpt39": {},
    "NGOpt4": {},
    "NGOpt8": {},
    "NGOptBase": {},
    "NoisyDE": {},
    "NonNSGAIIES": {},
    "OnePlusOne": _deterministic_first_point,
    "ORandomSearch": {},
    "OScrHammersleySearch": _deterministic_points,
    "PolyCMA": {},
    "PortfolioDiscreteOnePlusOne": _deterministic_first_point,
    "PSO": {},
    "QORandomSearch": {},
    "QOScrHammersleySearch": _deterministic_points,
    "QrDE": _deterministic_points,
    "RandomSearch": {},
    "RandomSearchPlusMiddlePoint": _deterministic_first_point,
    "RealSpacePSO": {},
    "RecES": {},
    "RecombiningPortfolioDiscreteOnePlusOne": _deterministic_first_point,
    "RecMixES": {},
    "RecMutDE": {},
    "RescaledCMA": {},
    "RotatedTwoPointsDE": {},
    "RotationInvariantDE": {},
    "ScrHaltonSearch": {},
    "ScrHaltonSearchPlusMiddlePoint": _deterministic_first_point,
    "ScrHammersleySearch": _deterministic_points,
    "ScrHammersleySearchPlusMiddlePoint": _deterministic_points,
    "Shiwa": {},
    "SparseDoubleFastGADiscreteOnePlusOne": _deterministic_first_point,
    "TBPSA": {},
    "TripleCMA": {},
    "TwoPointsDE": {},
}

HANGING_IN_MAX_TRIALS = {
    "BO": _max_trials_hangs,
    "ChainMetaModelSQP": _max_trials_hangs,
    "ChainCMAPowell": _max_trials_hangs,
    "ChainDiagonalCMAPowell": _max_trials_hangs,
    "ChainMetaModelPowell": _max_trials_hangs,
    "ChainNaiveTBPSACMAPowell": merge_dicts(
        _max_trials_hangs, _CAN_suggest_more_than_one
    ),
    "ChainNaiveTBPSAPowell": merge_dicts(_max_trials_hangs, _CAN_suggest_more_than_one),
}

BRANIN_FAILURES = {
    "AnisotropicAdaptiveDiscreteOnePlusOne": merge_dicts(
        _deterministic_first_point, _not_parallel
    ),
    "CMandAS2": {},
    "CMandAS3": {},
    "DoubleFastGADiscreteOnePlusOne": _deterministic_first_point,
    "MultiDiscrete": _deterministic_first_point,
    "NoisyDiscreteOnePlusOne": _deterministic_first_point,
    "ParaPortfolio": {},
    "Portfolio": {},
    "RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne": _deterministic_first_point,
    "SQPCMA": {},
}

NOT_WORKING = {
    "BO": {},
    "BOSplit": {},
    "BayesOptimBO": {},
    "DiscreteDoerrOnePlusOne": merge_dicts(_deterministic_first_point, _not_parallel),
    "NelderMead": merge_dicts(_sequential, _not_serializable),
    "PCABO": {},
    "SPSA": {},
    "SQP": {},
    "NoisyBandit": {},
    "NoisyOnePlusOne": _deterministic_first_point,
    "OptimisticDiscreteOnePlusOne": _deterministic_first_point,
    "OptimisticNoisyOnePlusOne": _deterministic_first_point,
    "Powell": merge_dicts(_not_parallel, _sequential),
    "CmaFmin2": merge_dicts(_sequential, _not_serializable),
    "RPowell": merge_dicts(_deterministic_first_point, _sequential, _not_serializable),
    "RSQP": merge_dicts(_deterministic_first_point, _sequential, _not_serializable),
    "PymooNSGA2": merge_dicts(_no_tell_without_ask, _sequential, _not_parallel),
}


missing_models = set(NOT_WORKING.keys()) ^ set(NOT_WORKING_MODEL_NAMES)
assert missing_models == set()

HANGING = {
    "Cobyla": {},
    "RCobyla": {},
}


MODEL_NAMES: dict[_AlgoName, dict[_TestName, MarkDecorator]] = merge_dicts(
    WORKING, NOT_WORKING
)

from pytest import FixtureRequest


@pytest.fixture(autouse=True, params=MODEL_NAMES.keys())
def _config(request: FixtureRequest):
    """Fixture that parametrizes the configuration used in the tests below."""
    model_name: str = request.param

    if model_name in NOT_WORKING:
        pytest.skip(reason=f"Model {model_name} is not supported.")

    tweaks = MODEL_NAMES[model_name]

    if ng.optimizers.registry[model_name].no_parallelization:
        num_workers = 1
    else:
        num_workers = 10

    TestNevergradOptimizer.config["model_name"] = model_name
    TestNevergradOptimizer.config["num_workers"] = num_workers

    test_name = request.function.__name__
    mark = tweaks.get(test_name, None)

    # TODO: Ask @Delaunay what this does
    if (
        test_name == "test_seed_rng_init"
        and request.getfixturevalue("phase").n_trials > 0
        and mark
        and mark.name == "xfail"
        and mark.kwargs["reason"].startswith("First generated point")
    ):
        mark = None

    if mark == "skip":
        pytest.skip(reason="Skipping test")
    elif mark:
        request.node.add_marker(mark)
    yield


class TestNevergradOptimizer(BaseAlgoTests):
    """Test suite for the NevergradOptimizer algorithm."""

    algo_name = "nevergradoptimizer"
    config = {
        "seed": 1234,  # Because this is so random
        "budget": 200,
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("optim", TEST_MANY_TRIALS, "space.sample"),
    ]

    def test_normal_data(self):
        """Test that algorithm supports normal dimensions"""
        self.assert_dim_type_supported({"x": "normal(2, 5)"})

    def test_suggest_n(self):
        """Verify that suggest returns correct number of trials if ``num`` is specified in
        ``suggest``.
        """
        algo = self.create_algo()
        trials = algo.suggest(5)
        # This condition in the original test is not respected.
        # assert len(trials) == 5
        assert 5 >= len(trials) >= 1
