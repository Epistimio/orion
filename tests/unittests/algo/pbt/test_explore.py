import numpy
import pytest
from base import RNGStub, TrialStub

from orion.algo.pbt.explore import PerturbExplore, PipelineExplore, ResampleExplore
from orion.algo.space import Categorical, Dimension
from orion.core.utils.flatten import flatten


class TestPipelineExplore:
    def test_no_explore(self):
        params = object()
        assert PipelineExplore([])(RNGStub(), None, params) is params

    def test_explore_otherwise_next(self):
        for i in range(4):
            explore = PipelineExplore(
                [
                    dict(of_type="explorestub", rval=None if j < i else i, some="args")
                    for j in range(4)
                ]
            )
            assert explore(RNGStub(), TrialStub(), None) == i

    def test_configuration(self):

        explore_configs = [
            dict(of_type="explorestub", some="args", rval=1, no_call=False),
            dict(of_type="explorestub", other="args", rval=None, no_call=True),
        ]
        explore = PipelineExplore(explore_configs)

        assert explore.configuration == dict(
            of_type="pipelineexplore", explore_configs=explore_configs
        )


class TestPerturb:
    @pytest.mark.parametrize("factor", [0.5, 1, 1.5])
    def test_perturb_real_factor(self, factor):
        explore = PerturbExplore(factor=factor)

        rng = RNGStub()
        rng.random = lambda: 1.0

        assert explore.perturb_real(rng, 1.0, (0.1, 2.0)) == factor

        rng.random = lambda: 0.0

        assert explore.perturb_real(rng, 1.0, (0.1, 2.0)) == 1.0 / factor

    def test_perturb_real_below_interval_cap(self):
        explore = PerturbExplore(factor=0.0, volatility=0)

        rng = RNGStub()
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_real(rng, 0.0, (1.0, 2.0)) == 1.0

        explore.volatility = 1000

        assert explore.perturb_real(rng, 0.0, (1.0, 2.0)) == 2.0

    def test_perturb_real_above_interval_cap(self):
        explore = PerturbExplore(factor=1.0, volatility=0)

        rng = RNGStub()
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_real(rng, 3.0, (1.0, 2.0)) == 2.0

        explore.volatility = 1000

        assert explore.perturb_real(rng, 3.0, (1.0, 2.0)) == 1.0

    @pytest.mark.parametrize("volatility", [0.0, 0.05, 1.0])
    def test_perturb_real_volatility_below(self, volatility):
        explore = PerturbExplore(factor=1.0, volatility=volatility)

        rng = RNGStub()
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_real(rng, 0.0, (1.0, 2.0)) == 1.0 + volatility

    @pytest.mark.parametrize("volatility", [0.0, 0.05, 1.0])
    def test_perturb_real_volatility_above(self, volatility):
        explore = PerturbExplore(factor=1.0, volatility=volatility)

        rng = RNGStub()
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_real(rng, 3.0, (1.0, 2.0)) == 2.0 - volatility

    @pytest.mark.parametrize("factor", [0.5, 0.75, 1, 1.5])
    def test_perturb_int_factor(self, factor):
        explore = PerturbExplore(factor=factor)

        rng = RNGStub()
        rng.random = lambda: 1.0

        assert explore.perturb_int(rng, 5, (0, 10)) == int(numpy.round(5 * factor))

        rng.random = lambda: 0.0

        assert explore.perturb_int(rng, 5, (0, 10)) == int(numpy.round(5 / factor))

    def test_perturb_int_duplicate_equal(self):
        explore = PerturbExplore(factor=1.0)

        rng = RNGStub()
        rng.random = lambda: 1.0

        assert explore.perturb_int(rng, 1, (0, 10)) == 1

    def test_perturb_int_no_duplicate_below(self):
        explore = PerturbExplore(factor=0.75)

        rng = RNGStub()
        rng.random = lambda: 1.0

        assert explore.perturb_int(rng, 1, (0, 10)) == 0

    def test_perturb_int_no_duplicate_above(self):
        explore = PerturbExplore(factor=0.75)

        rng = RNGStub()

        rng.random = lambda: 0.0

        assert explore.perturb_int(rng, 1, (0, 10)) == 2

    def test_perturb_int_no_out_of_bounds(self):
        explore = PerturbExplore(factor=0.75, volatility=0)

        rng = RNGStub()

        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_int(rng, 0, (0, 10)) == 0

        rng.random = lambda: 0.0
        rng.normal = lambda mean, variance: variance

        assert explore.perturb_int(rng, 10, (0, 10)) == 10

    def test_perturb_cat(self):
        explore = PerturbExplore()
        rng = RNGStub()
        rng.randint = lambda low, high=None, size=None: [1] if size else 1

        dim = Categorical("name", ["one", "two", 3, 4.0])
        assert explore.perturb_cat(rng, "whatever", dim) in dim

    def test_perturb(self, space):
        explore = PerturbExplore()
        rng = RNGStub()
        rng.randint = lambda low, high=None, size=None: numpy.ones(size) if size else 1
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: 0.0

        params = {"x": 1.0, "y": 2, "z": 0, "f": 10}
        new_params = explore(rng, space, params)
        for key in space.keys():
            assert new_params[key] in space[key]

    def test_perturb_hierarchical_params(self, hspace):
        explore = PerturbExplore()
        rng = RNGStub()
        rng.randint = lambda low, high=None, size=None: numpy.ones(size) if size else 1
        rng.random = lambda: 1.0
        rng.normal = lambda mean, variance: 0.0

        params = {"numerical": {"x": 1.0, "y": 2, "f": 10}, "z": 0}
        new_params = explore(rng, hspace, params)
        assert "numerical" in new_params
        assert "x" in new_params["numerical"]
        for key in hspace.keys():
            assert flatten(new_params)[key] in hspace[key]

    def test_perturb_with_invalid_dim(self, space, monkeypatch):
        explore = PerturbExplore()

        monkeypatch.setattr(Dimension, "type", "type_that_dont_exist")

        with pytest.raises(
            ValueError, match="Unsupported dimension type type_that_dont_exist"
        ):
            explore(RNGStub(), space, {"x": 1.0, "y": 2, "z": 0, "f": 10})

    def test_configuration(self):

        explore = PerturbExplore(factor=2.0, volatility=10.0)

        assert explore.configuration == dict(
            of_type="perturbexplore", factor=2.0, volatility=10.0
        )


class TestResample:
    def test_resample_probability(self, space):
        explore = ResampleExplore(probability=0.5)

        rng = RNGStub()
        rng.randint = lambda low, high, size: [1]
        rng.random = lambda: 0.5

        params = {"x": 1.0, "y": 2, "z": 0, "f": 10}

        assert explore(rng, space, params) is params

        rng.random = lambda: 0.4

        assert explore(rng, space, params) is not params

    def test_configuration(self):
        explore = ResampleExplore(probability=0.5)
        assert explore.configuration == dict(of_type="resampleexplore", probability=0.5)
