#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.space_builder`."""
import pytest
from scipy.stats import distributions as dists

from orion.algo.space import Categorical, Fidelity, Integer, Real
from orion.core.io.space_builder import DimensionBuilder, SpaceBuilder


@pytest.fixture(scope="module")
def dimbuilder():
    """Return a `DimensionBuilder` instance."""
    return DimensionBuilder()


@pytest.fixture(scope="module")
def spacebuilder():
    """Return a `SpaceBuilder` instance."""
    return SpaceBuilder()


class TestDimensionBuilder:
    """Ways of Dimensions builder."""

    def test_build_loguniform(self, dimbuilder):
        """Check that loguniform is built into reciprocal correctly."""
        dim = dimbuilder.build("yolo", "loguniform(0.001, 10)")
        assert isinstance(dim, Real)
        assert dim.name == "yolo"
        assert dim._prior_name == "reciprocal"
        assert 3.3 in dim and 11.1 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

        dim = dimbuilder.build("yolo2", "loguniform(1, 1000, discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == "yolo2"
        assert dim._prior_name == "reciprocal"
        assert 3 in dim and 0 not in dim and 3.3 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

    def test_eval_nonono(self, dimbuilder):
        """Make malevolent/naive eval access more difficult. I think."""
        with pytest.raises(RuntimeError):
            dimbuilder.build("la", "__class__")

    def test_build_a_good_real(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build("yolo2", "alpha(0.9, low=0, high=10, shape=2)")
        assert isinstance(dim, Real)
        assert dim.name == "yolo2"
        assert dim._prior_name == "alpha"
        assert 3.3 not in dim
        assert (3.3, 11.1) not in dim
        assert (3.3, 6) in dim
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_a_good_integer(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build("yolo3", "poisson(5)")
        assert isinstance(dim, Integer)
        assert dim.name == "yolo3"
        assert dim._prior_name == "poisson"
        assert isinstance(dim.prior, dists.poisson_gen)

    def test_build_a_good_real_discrete(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build("yolo3", "alpha(1.1, discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == "yolo3"
        assert dim._prior_name == "alpha"
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_a_good_fidelity(self, dimbuilder):
        """Check that a Fidelity dimension is correctly built."""
        dim = dimbuilder.build("epoch", "fidelity(1, 16, 4)")
        assert isinstance(dim, Fidelity)
        assert dim.name == "epoch"
        assert dim.low == 1
        assert dim.high == 16
        assert dim.base == 4
        assert dim._prior_name == "None"
        assert dim.prior is None

    def test_build_fidelity_default_base(self, dimbuilder):
        """Check that a Fidelity dimension is correctly built with default base."""
        dim = dimbuilder.build("epoch", "fidelity(1, 16)")
        assert isinstance(dim, Fidelity)
        assert dim.name == "epoch"
        assert dim.low == 1
        assert dim.high == 16
        assert dim.base == 2
        assert dim._prior_name == "None"
        assert dim.prior is None

    def test_build_fails_because_of_name(self, dimbuilder):
        """Build fails because distribution name is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo3", "lalala(1.1, discrete=True)")
        assert "Parameter" in str(exc.value)
        assert "supported" in str(exc.value)

    def test_build_fails_because_of_unexpected_args(self, dimbuilder):
        """Build fails because argument is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo3", "alpha(1.1, whatisthis=5, discrete=True)")
        assert "Parameter" in str(exc.value)
        assert "unexpected" in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_run(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo2", "alpha(0.9, low=5, high=6, shape=2)")
        assert "Parameter" in str(exc.value)
        assert "Improbable bounds" in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_init(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo2", "alpha(0.9, low=4, high=10, size=2)")
        assert "Parameter" in str(exc.value)
        assert "size" in str(exc.value.__cause__)

    def test_build_gaussian(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build("yolo", "gaussian(3, 5)")
        assert isinstance(dim, Real)
        assert dim.name == "yolo"
        assert dim._prior_name == "norm"
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build("yolo2", "gaussian(1, 0.5, discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == "yolo2"
        assert dim._prior_name == "norm"
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_normal(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build("yolo", "normal(0.001, 10)")
        assert isinstance(dim, Real)
        assert dim.name == "yolo"
        assert dim._prior_name == "norm"
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build("yolo2", "normal(1, 0.5, discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == "yolo2"
        assert dim._prior_name == "norm"
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_choices(self, dimbuilder):
        """Create correctly a `Categorical` dimension."""
        dim = dimbuilder.build("yolo", "choices('adfa', 1, 0.3, 'asaga', shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == "yolo"
        assert dim._prior_name == "Distribution"
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build("yolo", "choices(['adfa', 1])")
        assert isinstance(dim, Categorical)
        assert dim.name == "yolo"
        assert dim._prior_name == "Distribution"
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build("yolo2", "choices({'adfa': 0.1, 3: 0.4, 5: 0.5})")
        assert isinstance(dim, Categorical)
        assert dim.name == "yolo2"
        assert dim._prior_name == "Distribution"
        assert isinstance(dim.prior, dists.rv_discrete)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo2", "choices({'adfa': 0.1, 3: 0.4})")
        assert "Parameter" in str(exc.value)
        assert "sum" in str(exc.value.__cause__)

    def test_build_fails_because_empty_args(self, dimbuilder):
        """What happens if somebody 'forgets' stuff?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo", "choices()")
        assert "Parameter" in str(exc.value)
        assert "categories" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build("what", "alpha()")
        assert "Parameter" in str(exc.value)
        assert "positional" in str(exc.value.__cause__)

    def test_build_fails_because_troll(self, dimbuilder):
        """What happens if somebody does not fit regular expression expected?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build("yolo", "lalalala")
        assert "Parameter" in str(exc.value)
        assert "form for prior" in str(exc.value)


class TestSpaceBuilder:
    """Check whether space definition from various input format is successful."""

    def test_configuration_rebuild(self, spacebuilder):
        """Test that configuration can be used to recreate a space."""
        prior = {
            "x": "uniform(0, 10, discrete=True)",
            "y": "loguniform(1e-08, 1)",
            "z": "choices(['voici', 'voila', 2])",
        }
        space = spacebuilder.build(prior)
        assert space.configuration == prior

    def test_subdict_dimensions(self, spacebuilder):
        """Test space can have hierarchical structure."""
        prior = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)", "z": "choices(['voici', 'voila', 2])"},
        }
        space = spacebuilder.build(prior)
        assert len(space) == 3
        assert "a.x" in space
        assert "b.y" in space
        assert "b.z" in space
