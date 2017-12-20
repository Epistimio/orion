#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.algo.space`."""

from collections import OrderedDict

import numpy as np
from numpy.testing import assert_array_equal as assert_eq
import pytest
from scipy.stats import distributions as dists

from metaopt.algo.space import (Categorical, Dimension, Integer, Real, Space)


@pytest.fixture(scope='function')
def seed():
    """Return a fixed ``numpy.random.RandomState`` and gloal seed."""
    seed = 5
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    return rng


class TestDimension(object):
    """Test methods of a Dimension object."""

    def test_simple_instance(self, seed):
        """Test Dimension.__init__."""
        dim = Dimension('yolo', 'alpha', 0.9)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.alpha.rvs(0.9) == samples[0]

        assert dists.alpha.interval(1.0, 0.9) == dim.interval()
        assert dists.alpha.interval(0.5, 0.9) == dim.interval(0.5)

        assert 1.0 in dim

        assert str(dim) == "Dimension(name=yolo, prior={alpha: (0.9,), {}}, shape=())"

        assert dim.name == 'yolo'
        assert dim.type == 'dimension'
        assert dim.shape == ()

    def test_shaped_instance(self, seed):
        """Use shape keyword argument."""
        dim = Dimension('yolo', 'alpha', 0.9, shape=(3, 2))
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert_eq(dists.alpha.rvs(0.9, size=(3, 2)), samples[0])

        assert dim.shape == (3, 2)

        dim = Dimension('yolo', 'alpha', 0.9, shape=4)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert_eq(dists.alpha.rvs(0.9, size=4), samples[0])

        assert dim.shape == (4,)

    def test_bad_size_kwarg(self):
        """Should not be able to use 'size' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'alpha', 0.9, size=(3, 2))

    def test_bad_seed_kwarg(self):
        """Should not be able to use 'seed' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'alpha', 0.9, seed=8)

    def test_bad_rng_kwarg(self):
        """Should not be able to use 'random_state' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'alpha', 0.9, random_state=8)

    def test_with_predefined_dist(self, seed):
        """Use an already defined distribution object as prior arg."""
        dim = Dimension('yolo', dists.alpha, 0.9)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.alpha.rvs(0.9) == samples[0]

    def test_discrete_True(self, seed):
        """Use explicitly discrete argument == True."""
        dim = Dimension('yolo', 'uniform', -3, 4, shape=(4, 4), discrete=True)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        test_array = np.array([[-3, 0, -3, 0],
                               [-2, -1, 0, -1],
                               [-2, -3, -3, -1],
                               [-2, -3, 0, -2]], dtype=np.int)
        assert_eq(test_array, samples[0])

    def test_discrete_False(self, seed):
        """Use explicitly discrete argument == False."""
        dim = Dimension('yolo', 'uniform', -3, 4, shape=(4, 4), discrete=False)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert_eq(dists.uniform.rvs(-3, 4, size=(4, 4)), samples[0])

    def test_many_samples(self, seed):
        """More than 1."""
        dim = Dimension('yolo', 'uniform', -3, 4, shape=(4, 4))
        samples = dim.sample(n_samples=4, seed=seed)
        assert len(samples) == 4
        assert_eq(dists.uniform.rvs(-3, 4, size=(4, 4)), samples[0])

    def test_interval(self):
        """Test that bounds on variable."""
        dim = Dimension('yolo', 'uniform', -3, 4)
        assert dim.interval(1.0) == (-3.0, 1.0)  # reminder that `scale` is not upper bound

    def test_contains_bounds(self):
        """Test __contains__ for bounds."""
        dim = Dimension('yolo', 'uniform', -3, 4)
        assert -3 in dim
        assert -2 in dim
        assert 0 in dim
        assert 0.9999 in dim
        assert 1 not in dim

    def test_contains_shape(self):
        """Test __contains__ for shape check."""
        dim = Dimension(None, 'uniform', -3, 4, shape=(4, 4))
        assert -3 not in dim
        assert -2 not in dim
        assert 0 not in dim
        assert 0.9999 not in dim
        assert 1 not in dim
        assert dists.uniform.rvs(-3, 4, size=(4, 4)) in dim

    def test_set_bad_name(self):
        """Try setting a name other than str or None."""
        dim = Dimension('yolo', 'uniform', -3, 4, shape=(4, 4))
        with pytest.raises(TypeError):
            dim.name = 4


class TestReal(object):
    """Test methods of a `Real` object."""

    def test_as_you_are_as_you_were(self, seed):
        """Test Real.__init__."""
        dim = Real('yolo', 'alpha', 0.9)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.alpha.rvs(0.9) == samples[0]

        assert dists.alpha.interval(1.0, 0.9) == dim.interval()
        assert dists.alpha.interval(0.5, 0.9) == dim.interval(0.5)

        assert 1.0 in dim

        assert str(dim) == "Real(name=yolo, prior={alpha: (0.9,), {}}, shape=())"

        assert dim.name == 'yolo'
        assert dim.type == 'real'
        assert dim.shape == ()

    def test_contains_extra_bounds(self):
        """Test __contains__ for the extra bounds."""
        dim = Real('yolo', 'norm', 0, 3, low=-3, high=+3)
        assert dists.uniform.rvs(-3, 3) in dim
        assert -4 not in dim
        assert +4 not in dim
        assert (1, 2) not in dim

    def test_sample_from_extra_bounds_good(self):
        """Randomized test **successful** sampling with the extra bounds."""
        dim = Real('yolo', 'norm', 0, 2, low=-5, high=+5, shape=(4, 4))
        for _ in range(8):
            samples = dim.sample(8)
            for sample in samples:
                assert sample in dim

    def test_sample_from_extra_bounds_bad(self):
        """Randomized test **unsuccessfully** sampling with the extra bounds."""
        dim = Real('yolo', 'norm', 0, 2, low=-2, high=+2, shape=(4, 4))
        with pytest.raises(ValueError) as exc:
            for _ in range(8):
                samples = dim.sample(8)
                for sample in samples:
                    assert sample in dim
        assert "Improbable bounds" in str(exc.value)

    def test_bad_bounds(self):
        """Try setting bound with high <= low."""
        with pytest.raises(ValueError):
            Real('yolo', 'norm', 0, 2, low=+2, high=-2, shape=(4, 4))
        with pytest.raises(ValueError):
            Real('yolo', 'norm', 0, 2, low=+2, high=+2, shape=(4, 4))


class TestInteger(object):
    """Test methods of a `Integer` object."""

    def test_as_you_are_as_you_were(self, seed):
        """Test Integer.__init__."""
        dim = Integer('yolo', 'uniform', -3, 6)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert samples[0] == -2

        assert dists.uniform.interval(1.0, -3, 6) == dim.interval()
        assert dists.uniform.interval(0.5, -3, 6) == dim.interval(0.5)

        assert 1.0 in dim

        assert str(dim) == "Integer(name=yolo, prior={uniform: (-3, 6), {}}, shape=())"

        assert dim.name == 'yolo'
        assert dim.type == 'integer'
        assert dim.shape == ()

    def test_contains(self):
        """Check for integer test."""
        dim = Integer('yolo', 'uniform', -3, 6)

        assert 0.1 not in dim
        assert (0.1, -0.2) not in dim
        assert 0 in dim
        assert (1, 2) not in dim
        assert 6 not in dim
        assert -3 in dim
        assert -4 not in dim


class TestCategorical(object):
    """Test methods of a `Categorical` object."""

    def test_with_tuple(self, seed):
        """Test Categorical.__init__ with a tuple."""
        categories = ('asdfa', 2)
        dim = Categorical('yolo', categories)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert samples[0] == 'asdfa'
        assert dim._probs == (0.5, 0.5)

        assert categories == dim.interval()
        assert categories == dim.interval(0.5)

        assert 2 in dim
        assert 3 not in dim

        assert str(dim) == "Categorical(name=yolo, prior=[('asdfa', 0.5), (2, 0.5)], shape=())"

        assert dim.name == 'yolo'
        assert dim.type == 'categorical'
        assert dim.shape == ()

    def test_with_dict(self, seed):
        """Test Categorical.__init__ with a dictionary."""
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)))
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert samples[0] == 2
        assert dim._probs == probs

        assert categories == dim.interval()
        assert categories == dim.interval(0.5)

        assert 2 in dim
        assert 0 not in dim

        assert dim.name == 'yolo'
        assert dim.type == 'categorical'
        assert dim.shape == ()

    def test_contains_wrong_shape(self):
        """Check correct category but wrongly shaped array."""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        dim = Categorical('yolo', categories, shape=2)

        assert 3 not in dim
        assert ('asdfa', 2) in dim

    def test_repr_too_many_cats(self):
        """Check ellipsis on str/repr of too many categories."""
        categories = tuple(range(10))
        dim = Categorical('yolo', categories, shape=2)

        assert str(dim) == "Categorical(name=yolo, " \
                           "prior=[(0, 0.1), (1, 0.1), ..., (8, 0.1), (9, 0.1)], " \
                           "shape=(2,))"

    def test_bad_probabilities(self):
        """User provided bad probabilities."""
        categories = {'asdfa': 0.05, 2: 0.2, 3: 0.3, 4: 0.4}
        with pytest.raises(ValueError):
            Categorical('yolo', categories, shape=2)


class TestSpace(object):
    """Test methods of a `Space` object."""

    def test_init(self):
        """Instantiate space, must be an ordered dictionary."""
        space = Space()
        assert isinstance(space, OrderedDict)

    def test_register_and_contain(self):
        """Register bunch of dimensions, check if points/name are in space."""
        space = Space()

        assert 'yolo' not in space
        assert (('asdfa', 2), 0, 3.5) not in space

        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        dim = Categorical('yolo', categories, shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        dim = Real('yolo3', 'alpha', 0.9)
        space.register(dim)

        assert 'yolo' in space
        assert 'yolo2' in space
        assert 'yolo3' in space

        assert (('asdfa', 2), 0, 3.5) in space
        assert (('asdfa', 2), 7, 3.5) not in space

    def test_bad_contain(self):
        """Checking with no iterables does no good."""
        space = Space()
        with pytest.raises(TypeError):
            5 in space

    def test_sample(self, seed):
        """Check whether sampling works correctly."""
        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        dim = Real('yolo3', 'alpha', 0.9)
        space.register(dim)

        point = space.sample(seed=seed)
        print(point)

        points = space.sample(2, seed=seed)
        print(points)

    def test_interval(self):
        """Check whether interval is cool."""
        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        dim = Real('yolo3', 'alpha', 0.9)
        space.register(dim)

        assert space.interval() == [categories, (-3, 3), (0, np.inf)]

    def test_bad_setitem(self):
        """Check exceptions in setting items in Space."""
        space = Space()

        with pytest.raises(TypeError) as exc:
            space[5] = Integer('yolo', 'uniform', -3, 6)
        assert "string" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            space['ispis'] = 'nope'
        assert "Dimension" in str(exc.value)

        space.register(Integer('yolo', 'uniform', -3, 6))
        with pytest.raises(ValueError) as exc:
            space.register(Real('yolo', 'uniform', 0, 6))
        assert "another name" in str(exc.value)

    def test_getitem(self):
        """Test getting dimensions from space."""
        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        dim = Real('yolo3', 'alpha', 0.9)
        space.register(dim)

        assert space['yolo'].type == 'categorical'
        assert space[0].type == 'categorical'

        with pytest.raises(KeyError):
            space['asdf']

        with pytest.raises(IndexError):
            space[3]

    def test_repr(self):
        """Test str/repr."""
        space = Space()
        dim = Integer('yolo2', 'uniform', -3, 6, shape=(2,))
        space.register(dim)
        dim = Real('yolo3', 'alpha', 0.9)
        space.register(dim)

        assert str(space) == "Space(["\
                             "Integer(name=yolo2, prior={uniform: (-3, 6), {}}, shape=(2,)),\n" \
                             "       Real(name=yolo3, prior={alpha: (0.9,), {}}, shape=())])"
