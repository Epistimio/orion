#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.space`."""

from collections import (defaultdict, OrderedDict)

import numpy as np
from numpy.testing import assert_array_equal as assert_eq
import pytest
from scipy.stats import distributions as dists

from orion.algo.space import (
    Categorical, check_random_state, Dimension, Fidelity, Integer, Real, Space)


class TestCheckRandomState:
    """Test `orion.algo.space.check_random_state`"""

    def test_rng_null(self):
        """Test that passing None returns numpy._rand"""
        assert check_random_state(None) is np.random.mtrand._rand

    def test_rng_random_state(self):
        """Test that passing RandomState returns itself"""
        rng = np.random.RandomState(1)
        assert check_random_state(rng) is rng

    def test_rng_int(self):
        """Test that passing int returns RandomState"""
        rng = check_random_state(1)
        assert isinstance(rng, np.random.RandomState)
        assert rng is not np.random.mtrand._rand

    def test_rng_tuple(self):
        """Test that passing int returns RandomState"""
        rng = check_random_state((1, 12, 123))
        assert isinstance(rng, np.random.RandomState)
        assert rng is not np.random.mtrand._rand

    def test_rng_invalid_value(self):
        """Test that passing int returns RandomState"""
        with pytest.raises(ValueError) as exc:
            check_random_state('oh_no_oh_no')

        assert '\'oh_no_oh_no\' cannot be used to seed' in str(exc.value)


class TestDimension(object):
    """Test methods of a Dimension object."""

    def test_simple_instance(self, seed):
        """Test Dimension.__init__."""
        dim = Dimension('yolo', 'norm', 0.9, 0.1)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.norm.rvs(0.9, 0.1) == samples[0]

        assert dists.norm.interval(1.0, 0.9, 0.1) == dim.interval()
        assert dists.norm.interval(0.5, 0.9, 0.1) == dim.interval(0.5)

        assert str(dim) == "Dimension(name=yolo, prior={norm: (0.9, 0.1), {}}, " \
                           "shape=(), default value=None)"

        assert dim.name == 'yolo'
        assert dim.type == 'dimension'
        assert dim.shape == ()

    def test_shaped_instance(self, seed):
        """Use shape keyword argument."""
        dim = Dimension('yolo', 'norm', 0.9, shape=(3, 2))
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert_eq(dists.norm.rvs(0.9, size=(3, 2)), samples[0])

        assert dim.shape == (3, 2)

        dim = Dimension('yolo', 'norm', 0.9, shape=4)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert_eq(dists.norm.rvs(0.9, size=4), samples[0])

        assert dim.shape == (4,)

    def test_ban_size_kwarg(self):
        """Should not be able to use 'size' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'norm', 0.9, size=(3, 2))

    def test_ban_seed_kwarg(self):
        """Should not be able to use 'seed' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'norm', 0.9, seed=8)

    def test_ban_rng_kwarg(self):
        """Should not be able to use 'random_state' kwarg."""
        with pytest.raises(ValueError):
            Dimension('yolo', 'norm', 0.9, random_state=8)

    def test_with_predefined_dist(self, seed):
        """Use an already defined distribution object as prior arg."""
        dim = Dimension('yolo', dists.norm, 0.9)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.norm.rvs(0.9) == samples[0]

    def test_ban_discrete_kwarg(self):
        """Do not allow use for 'discrete' kwarg, because now there's `_Discrete`."""
        with pytest.raises(ValueError) as exc:
            Dimension('yolo', 'uniform', -3, 4, shape=(4, 4), discrete=True)
        assert "pure `_Discrete`" in str(exc.value)

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
        with pytest.raises(NotImplementedError):
            assert -3 in dim

    def test_contains_shape(self):
        """Test __contains__ for shape check."""
        dim = Dimension(None, 'uniform', -3, 4, shape=(4, 4))

        with pytest.raises(NotImplementedError):
            assert dists.uniform.rvs(-3, 4, size=(4, 4)) in dim

    def test_set_bad_name(self):
        """Try setting a name other than str or None."""
        dim = Dimension('yolo', 'uniform', -3, 4, shape=(4, 4))
        with pytest.raises(TypeError):
            dim.name = 4

    def test_init_with_default_value(self):
        """Make sure the __contains__ method does not work"""
        with pytest.raises(NotImplementedError):
            Dimension('yolo', 'uniform', -3, 4, default_value=4)

    def test_no_default_value(self):
        """Test that no giving a default value assigns None"""
        dim = Dimension('yolo', 'uniform', -3, 4)
        assert dim.default_value is None

    def test_no_prior(self):
        """Test that giving a null prior defaults prior_name to `None`."""
        dim = Dimension('yolo', None)
        print(dim._prior_name)
        assert dim.prior is None
        assert dim._prior_name is 'None'


class TestReal(object):
    """Test methods of a `Real` object."""

    def test_simple_instance(self, seed):
        """Test Real.__init__."""
        dim = Real('yolo', 'norm', 0.9)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert dists.norm.rvs(0.9) == samples[0]

        assert dists.norm.interval(1.0, 0.9) == dim.interval()
        assert dists.norm.interval(0.5, 0.9) == dim.interval(0.5)

        assert 1.0 in dim

        assert str(dim) == "Real(name=yolo, prior={norm: (0.9,), {}}, shape=(), default value=None)"
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
            dim.sample(8)
        assert "Improbable bounds" in str(exc.value)

    def test_bad_bounds(self):
        """Try setting bound with high <= low."""
        with pytest.raises(ValueError):
            Real('yolo', 'norm', 0, 2, low=+2, high=-2, shape=(4, 4))
        with pytest.raises(ValueError):
            Real('yolo', 'norm', 0, 2, low=+2, high=+2, shape=(4, 4))

    def test_interval(self):
        """Interval takes into account explicitly bounds."""
        dim = Real('yolo', 'norm', 0, 3, low=-3, high=+3)
        assert dim.interval() == (-3, 3)

        dim = Real('yolo', 'alpha', 0.9, low=-3, high=+3)
        assert dim.interval() == (0, 3)

        dim = Real('yolo', 'uniform', -2, 4, low=-3, high=+3)
        assert dim.interval() == (-2.0, 2.0)

    def test_init_with_default_value(self):
        """Make sure the default value is set"""
        dim = Real('yolo', 'uniform', -3, 10, default_value=2.0)

        assert type(dim.default_value) is float

    def test_set_outside_bounds_default_value(self):
        """Make sure default value is inside the bounds"""
        with pytest.raises(ValueError):
            Real('yolo', 'uniform', -3, 2, default_value=5)

    def test_no_default_value(self):
        """Make sure the default value is None"""
        dim = Real('yolo', 'uniform', -3, 4)
        assert dim.default_value is None

    def test_cast_list(self):
        """Make sure list are cast to float and returned as list of values"""
        dim = Real('yolo', 'uniform', -3, 4)
        assert dim.cast(['1', '2']) == [1.0, 2.0]

    def test_cast_array(self):
        """Make sure array are cast to float and returned as array of values"""
        dim = Real('yolo', 'uniform', -3, 4)
        assert np.all(dim.cast(np.array(['1', '2'])) == np.array([1.0, 2.0]))


class TestInteger(object):
    """Test methods of a `Integer` object."""

    def test_simple_instance(self, seed):
        """Test Integer.__init__."""
        dim = Integer('yolo', 'uniform', -3, 6)
        samples = dim.sample(seed=seed)
        assert len(samples) == 1
        assert samples[0] == -2

        assert dim.interval() == (-3, 3)
        assert dim.interval(0.5) == (-2, 2)

        assert 1.0 in dim

        assert str(dim) == "Integer(name=yolo, prior={uniform: (-3, 6), {}}, "\
                           "shape=(), default value=None)"

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

    def test_interval_with_infs(self):
        """Regression test: Interval handles correctly extreme bounds."""
        dim = Integer('yolo', 'poisson', 5)
        # XXX: Complete this on both end of interval when scipy bug is fixed
        assert dim.interval()[1] == np.inf

    @pytest.mark.xfail(reason="scipy bug")
    def test_scipy_integer_dist_interval_bug(self):
        """Scipy does not return the correct answer for integer distributions."""
        dim = Integer('yolo', 'randint', -3, 6)
        assert dim.interval() == (-3, 6)
        assert dim.interval(1.0) == (-3, 6)
        assert dim.interval(0.9999) == (-3, 6)

        dim = Integer('yolo2', 'randint', -2, 4, loc=8)
        assert dim.interval() == (6, 12)

    def test_init_with_default_value(self):
        """Make sure the type of the default value is int"""
        dim = Integer('yolo', 'uniform', -3, 10, default_value=2)

        assert type(dim.default_value) is int

    def test_set_outside_bounds_default_value(self):
        """Make sure the default value is inside the bounds of the dimensions"""
        with pytest.raises(ValueError):
            Integer('yolo', 'uniform', -3, 2, default_value=4)

    def test_no_default_value(self):
        """Make sure the default value is None"""
        dim = Integer('yolo', 'uniform', -3, 4)
        assert dim.default_value is None

    def test_cast_list(self):
        """Make sure list are cast to int and returned as list of values"""
        dim = Integer('yolo', 'uniform', -3, 4)
        assert dim.cast(['1', '2']) == [1, 2]

    def test_cast_array(self):
        """Make sure array are cast to int and returned as array of values"""
        dim = Integer('yolo', 'uniform', -3, 4)
        assert np.all(dim.cast(np.array(['1', '2'])) == np.array([1, 2]))


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

        assert categories == dim.categories

        assert 2 in dim
        assert 3 not in dim

        assert str(dim) == "Categorical(name=yolo, prior={asdfa: 0.50, 2: 0.50}, "\
                           "shape=(), default value=None)"

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

        assert categories == dim.categories

        assert 2 in dim
        assert 0 not in dim

        assert dim.name == 'yolo'
        assert dim.type == 'categorical'
        assert dim.shape == ()

    def test_probabilities_are_ok(self, seed):
        """Test that the probabilities given are legit using law of big numbers."""
        bins = defaultdict(int)
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', '2', '3', '4')
        categories = OrderedDict(zip(categories, probs))
        dim = Categorical('yolo', categories)
        for _ in range(500):
            sample = dim.sample(seed=seed)[0]
            bins[sample] += 1
        for keys in bins.keys():
            bins[keys] /= float(500)
        for key, value in categories.items():
            assert abs(bins[key] - value) < 0.01

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
                           "prior={0: 0.10, 1: 0.10, ..., 8: 0.10, 9: 0.10}, " \
                           "shape=(2,), default value=None)"

    def test_bad_probabilities(self):
        """User provided bad probabilities."""
        categories = {'asdfa': 0.05, 2: 0.2, 3: 0.3, 4: 0.4}
        with pytest.raises(ValueError):
            Categorical('yolo', categories, shape=2)

    def test_interval_is_banned(self):
        """Check that calling `Categorical.interval` raises `RuntimeError`."""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        dim = Categorical('yolo', categories, shape=2)

        with pytest.raises(RuntimeError) as exc:
            dim.interval()
        assert 'not ordered' in str(exc.value)

    def test_that_objects_types_are_ok(self):
        """Check that output samples are of the correct type.

        Don't let numpy mess with their automatic type inference.
        """
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 'lalala': 0.4}
        dim = Categorical('yolo', categories)

        assert '2' not in dim
        assert 2 in dim
        assert 'asdfa' in dim

        dim = Categorical('yolo', categories, shape=(2,))

        assert ['2', 'asdfa'] not in dim
        assert [2, 'asdfa'] in dim

    def test_init_with_default_value_string(self):
        """Make sure the default value is of the correct type"""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 'lalala': 0.4}
        dim = Categorical('yolo', categories, default_value="asdfa")

        assert type(dim.default_value) is str

    def test_init_with_default_value_int(self):
        """Make sure the default value is of the correct type"""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 'lalala': 0.4}
        dim = Categorical('yolo', categories, default_value=2)

        assert type(dim.default_value) is int

    def test_init_with_wrong_default_value(self):
        """Make sure the default value exists"""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 'lalala': 0.4}

        with pytest.raises(ValueError):
            Categorical('yolo', categories, default_value=2.3)

    def test_no_default_value(self):
        """Make sure the default value is None"""
        categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 'lalala': 0.4}
        dim = Categorical('yolo', categories)
        assert dim.default_value is None

    def test_cast_list(self):
        """Make sure list are cast to categories and returned as list"""
        categories = {'asdfa': 0.1, 2: 0.2, 3.0: 0.3, 'lalala': 0.4}
        dim = Categorical('yolo', categories)
        assert dim.cast(['asdfa']) == ['asdfa']
        assert dim.cast(['2']) == [2]
        assert dim.cast(['3.0']) == [3.0]

    def test_cast_list_multidim(self):
        """Make sure array are cast to int and returned as array of values"""
        categories = list(range(10))
        categories[0] = 'asdfa'
        categories[2] = 'lalala'
        dim = Categorical('yolo', categories, shape=2)
        sample = ['asdfa', '1']  # np.array(['asdfa', '1'], dtype=np.object)
        assert dim.cast(sample) == ['asdfa', 1]

    def test_cast_array_multidim(self):
        """Make sure array are cast to int and returned as array of values"""
        categories = list(range(10))
        categories[0] = 'asdfa'
        categories[2] = 'lalala'
        dim = Categorical('yolo', categories, shape=2)
        sample = np.array(['asdfa', '1'], dtype=np.object)
        assert np.all(dim.cast(sample) == np.array(['asdfa', 1], dtype=np.object))

    def test_cast_bad_category(self):
        """Make sure array are cast to int and returned as array of values"""
        categories = list(range(10))
        dim = Categorical('yolo', categories, shape=2)
        sample = np.array(['asdfa', '1'], dtype=np.object)
        with pytest.raises(ValueError) as exc:
            dim.cast(sample)
        assert "Invalid category: asdfa" in str(exc.value)


class TestFidelity(object):
    """Test methods of a Fidelity object."""

    def test_simple_instance(self):
        """Test Fidelity.__init__."""
        dim = Fidelity('epoch')

        assert str(dim) == "Fidelity(name=epoch)"
        assert dim.name == 'epoch'
        assert dim.type == 'fidelity'
        assert dim.shape is None

    def test_sampling(self):
        """Make sure Fidelity simply returns `fidelity`"""
        dim = Fidelity('epoch')

        assert dim.sample() == ['fidelity']

    def test_contains(self):
        """Make sure fidelity.__contains__ always returns True"""
        dim = Fidelity('epoch')

        assert None in dim
        assert 0 in dim
        assert object() in dim

    def test_interval(self):
        """Check that error is being raised."""
        dim = Fidelity('epoch')
        with pytest.raises(NotImplementedError):
            dim.interval()

    def test_cast(self):
        """Check that error is being raised."""
        dim = Fidelity('epoch')
        with pytest.raises(NotImplementedError):
            dim.cast()


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
        dim = Real('yolo3', 'norm', 0.9)
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
        dim1 = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=(2, 2))
        space.register(dim1)
        dim2 = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim2)
        dim3 = Real('yolo3', 'norm', 0.9)
        space.register(dim3)

        point = space.sample(seed=seed)
        test_point = [(dim1.sample()[0],
                       dim2.sample()[0],
                       dim3.sample()[0]), ]
        assert len(point) == len(test_point) == 1
        assert len(point[0]) == len(test_point[0]) == 3
        assert np.all(point[0][0] == test_point[0][0])
        assert point[0][1] == test_point[0][1]
        assert point[0][2] == test_point[0][2]

        points = space.sample(2, seed=seed)
        points1 = dim1.sample(2)
        points2 = dim2.sample(2)
        points3 = dim3.sample(2)
        test_points = [(points1[0], points2[0], points3[0]),
                       (points1[1], points2[1], points3[1])]
        assert len(points) == len(test_points) == 2
        for i in range(2):
            assert len(points[i]) == len(test_points[i]) == 3
            assert np.all(points[i][0] == test_points[i][0])
            assert points[i][1] == test_points[i][1]
            assert points[i][2] == test_points[i][2]

    def test_interval(self):
        """Check whether interval is cool."""
        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        dim = Real('yolo3', 'norm', 0.9)
        space.register(dim)

        assert space.interval() == [categories, (-3, 3), (-np.inf, np.inf)]

    def test_bad_setitem(self):
        """Check exceptions in setting items in Space."""
        space = Space()

        # The name of an integer must be a of `str` type.
        # Integers are reversed for indexing the OrderedDict.
        with pytest.raises(TypeError) as exc:
            space[5] = Integer('yolo', 'uniform', -3, 6)
        assert "string" in str(exc.value)

        # Only object of type `Dimension` are allowed in `Space`.
        with pytest.raises(TypeError) as exc:
            space['ispis'] = 'nope'
        assert "Dimension" in str(exc.value)

        # Cannot register something with the same name.
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
        dim = Real('yolo3', 'norm', 0.9)
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
        dim = Real('yolo3', 'norm', 0.9)
        space.register(dim)

        assert str(space) == "Space(["\
                             "Integer(name=yolo2, prior={uniform: (-3, 6), {}}, shape=(2,), "\
                             "default value=None),\n"\
                             "       Real(name=yolo3, prior={norm: (0.9,), {}}, shape=(), "\
                             "default value=None)])"
