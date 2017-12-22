#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.core.io.space_builder`."""

import sys

import pytest
from scipy.stats import distributions as dists

from metaopt.algo.space import (Categorical, Integer, Real)
from metaopt.core.io.space_builder import DimensionBuilder


@pytest.fixture(scope='module')
def dimbuilder():
    """Return a `DimensionBuilder` instance."""
    return DimensionBuilder()


class TestDimensionBuilder(object):
    """Ways of Dimensions builder."""

    def test_build_loguniform(self, dimbuilder):
        """Check that loguniform is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'loguniform(0.001, 10)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'reciprocal'
        assert 3.3 in dim and 11.1 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

        dim = dimbuilder.build('yolo2', 'loguniform(1, 1000, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'reciprocal'
        assert 3 in dim and 0 not in dim and 3.3 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

    def test_eval_nonono(self, dimbuilder):
        """Make malevolent/naive eval access more difficult. I think."""
        with pytest.raises(RuntimeError):
            dimbuilder.build('la', "__class__")

    def test_build_a_good_real(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo2', 'alpha(0.9, low=0, high=10, shape=2)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'alpha'
        assert 3.3 not in dim and (3.3, 11.1) not in dim and (3.3, 6) in dim
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_a_good_integer(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo3', 'poisson(5)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo3'
        assert dim._prior_name == 'poisson'
        assert isinstance(dim.prior, dists.poisson_gen)

    def test_build_a_good_real_discrete(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo3', 'alpha(1.1, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo3'
        assert dim._prior_name == 'alpha'
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_fails_because_of_name(self, dimbuilder):
        """Build fails because distribution name is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo3', 'lalala(1.1, discrete=True)')
        assert 'Parameter' in str(exc.value)
        assert 'supported' in str(exc.value)

    def test_build_fails_because_of_unexpected_args(self, dimbuilder):
        """Build fails because argument is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo3', 'alpha(1.1, whatisthis=5, discrete=True)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'unexpected' in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_run(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', 'alpha(0.9, low=4, high=10, shape=2)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'Improbable bounds' in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_init(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', 'alpha(0.9, low=4, high=10, size=2)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'size' in str(exc.value.__cause__)

    def test_build_gaussian(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'gaussian(3, 5)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build('yolo2', 'gaussian(1, 0.5, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_normal(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'normal(0.001, 10)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build('yolo2', 'normal(1, 0.5, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_enum(self, dimbuilder):
        """Create correctly a `Categorical` dimension."""
        dim = dimbuilder.build('yolo', "enum('adfa', 1, 0.3, 'asaga', shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "enum(['adfa', 1])")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo2', "enum({'adfa': 0.1, 3: 0.4, 5: 0.5})")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', "enum({'adfa': 0.1, 3: 0.4})")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'sum' in str(exc.value.__cause__)

    def test_build_fails_because_empty_args(self, dimbuilder):
        """What happens if somebody 'forgets' stuff?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo', "enum()")
        assert 'Parameter' in str(exc.value)
        assert 'categories' in str(exc.value)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('what', "alpha()")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'positional' in str(exc.value.__cause__)

    def test_build_fails_because_troll(self, dimbuilder):
        """What happens if somebody does not fit regular expression expected?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo', "lalalala")
        assert 'Parameter' in str(exc.value)
        assert 'form for prior' in str(exc.value)

    def test_build_random(self, dimbuilder):
        """Things built by random keyword."""
        dim = dimbuilder.build('yolo', "random('adfa', 1, 0.3, 'asaga', shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(-3, 1, 8, 50, shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(['adfa', 1])")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(('adfa', 1))")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)
        assert dim.interval() == (('adfa', 1),)

        dim = dimbuilder.build('yolo2', "random({'adfa': 0.1, 3: 0.4, 5: 0.5})")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', "random({'adfa': 0.1, 3: 0.4})")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'sum' in str(exc.value.__cause__)

        dim = dimbuilder.build('yolo', "random('adfa', 1, shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(-3, 5.4, shape=4)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (-3, 5.4)

        dim = dimbuilder.build('yolo', "random(4)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (4.0, 5.0)

        dim = dimbuilder.build('yolo', "random()")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (0.0, 1.0)

        dim = dimbuilder.build('yolo', "random(-3, 5, shape=4, discrete=True)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (-3, 5)

        # These two below, do not make any sense for integers.. meh
        dim = dimbuilder.build('yolo', "random(4, discrete=True)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (4.0, 5.0)

        dim = dimbuilder.build('yolo', "random(discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (0.0, 1.0)
