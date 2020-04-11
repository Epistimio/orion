#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.tpe`."""

import numpy
import pytest

from orion.algo.space import Integer, Real, Space
from orion.algo.tpe import adaptive_parzen_estimator, compute_max_ei_point, GMMSampler, TPE


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Real('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', -2, 4)
    space.register(dim2)

    return space


@pytest.fixture
def tpe(space):
    """Return an instance of TPE."""
    return TPE(space)


@pytest.fixture
def gmm_sampler(tpe):
    """Return an instance of GMMSampler"""
    mus = numpy.linspace(-3, 3, num=12, endpoint=False)
    sigmas = [0.5] * 12

    return GMMSampler(tpe, mus, sigmas, -3, 3)


def test_compute_max_ei_point():
    """Test that max ei point is computed correctly"""
    points = numpy.linspace(-3, 3, num=10)
    below_likelis = numpy.linspace(0.5, 0.9, num=10)
    above_likes = numpy.linspace(0.2, 0.5, num=10)
    max_ei_point = compute_max_ei_point(points, below_likelis, above_likes)
    assert max_ei_point == 3


def test_adaptive_parzen_normal_estimator():
    """Test adaptive kernel estimator"""
    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    low = -1
    high = 5
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas == 6 / 10)

    # equal weight
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=True, flat_num=25)
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas == 6 / 10)

    # priori weight
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=0.5,
                                                     equal_weight=False, flat_num=25)
    priori_pos = numpy.searchsorted(mus, 2)
    assert weights[priori_pos] == 0.5 / 31
    assert numpy.all(weights[31 - priori_pos:] == weights[-1])
    assert numpy.all(sigmas == 6 / 10)

    # sigma clip
    obs_mus = numpy.linspace(-1, 5, num=8, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)
    assert numpy.all(sigmas == 6 / 8)


class TestGMMSampler():
    """Tests for TPE GMM Sampler"""

    def test_gmm_sampler_creation(self, gmm_sampler):
        """Test GMMSampler creation"""
        assert len(gmm_sampler.weights) == 12
        assert len(gmm_sampler.pdfs) == 12

    def test_sample(self, gmm_sampler):
        """Test GMMSampler sample function"""
        points = gmm_sampler.sample(25)
        points = numpy.array(points)

        assert len(points) <= 25
        assert numpy.all(points >= -3)
        assert numpy.all(points < 3)

    def test_get_loglikelis(self, gmm_sampler):
        """Test to get log likelis of points"""
        points = numpy.random.random(25)
        likelis = gmm_sampler.get_loglikelis(points)

        assert len(likelis) == len(points)


class TestTPE():
    """Tests for the algo TPE."""

    def test_seed_rng(self, tpe):
        """Test that algo is seeded properly"""
        tpe.seed_rng(1)
        a = tpe.suggest(1)[0]
        assert not numpy.allclose(a, tpe.suggest(1)[0])

        tpe.seed_rng(1)
        assert numpy.allclose(a, tpe.suggest(1)[0])

    def test_set_state(self, tpe):
        """Test that state is reset properly"""
        tpe.seed_rng(1)
        state = tpe.state_dict
        a = tpe.suggest(1)[0]
        assert not numpy.allclose(a, tpe.suggest(1)[0])

        tpe.set_state(state)
        assert numpy.allclose(a, tpe.suggest(1)[0])

    def test_split_trials(self, tpe):
        """Test observed trails can be split based on TPE gamma"""
        space = Space()
        dim1 = Real('yolo1', 'uniform', -3, 6)
        space.register(dim1)

        tpe.space = space
        tpe.gamma = 0.25

        points = numpy.linspace(-3, 3, num=10, endpoint=False)
        results = numpy.linspace(0, 1, num=10, endpoint=False)
        for point, result in zip(points, results):
            tpe.observe([[point]], [{'objective': result}])

        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4], [-1.8]]
        assert len(above_points) == 7

        tpe.gamma = 0.2
        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4]]
        assert len(above_points) == 8

    def test_unsupported_space(self):
        """Test tpe only work for supported search space"""
        space = Space()
        dim = Integer('yolo1', 'uniform', -2, 4)
        space.register(dim)

        with pytest.raises(ValueError) as ex:
            TPE(space)

        assert 'TPE now only supports Real Dimension' in str(ex.value)

        space = Space()
        dim = Real('yolo1', 'norm', 0.9)
        space.register(dim)

        with pytest.raises(ValueError) as ex:
            TPE(space)

        assert 'TPE now only supports uniform as prior' in str(ex.value)

        space = Space()
        dim = Real('yolo1', 'uniform', 0.9, shape=(2, 1))
        space.register(dim)

        with pytest.raises(ValueError) as ex:
            TPE(space)

        assert 'TPE now only supports 1D shape' in str(ex.value)

    def test_suggest(self, tpe):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(10):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 2
            assert not isinstance(point[0][0], tuple)
            tpe.observe(point, [{'objective': results[i]}])

        point = tpe.suggest(1)
        assert len(point) == 1
        assert len(point[0]) == 2
        assert not isinstance(point[0][0], tuple)

    def test_1d_shape(self, tpe):
        """Test suggest with 1D shape dimensions"""
        space = Space()
        dim1 = Real('yolo1', 'uniform', -3, 6, shape=(2))
        space.register(dim1)
        dim2 = Real('yolo2', 'uniform', -2, 4)
        space.register(dim2)

        tpe.space = space

        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(10):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 2
            assert len(point[0][0]) == 2
            tpe.observe(point, [{'objective': results[i]}])

        point = tpe.suggest(1)
        assert len(point) == 1
        assert len(point[0]) == 2
        assert len(point[0][0]) == 2
