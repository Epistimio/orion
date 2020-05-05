#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.tpe`."""

import numpy
import pytest
from scipy.stats import norm

from orion.algo.space import Integer, Real, Space
from orion.algo.tpe import adaptive_parzen_estimator, compute_max_ei_point, GMMSampler, TPE

numpy.random.seed(1)


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Real('yolo1', 'uniform', -10, 20)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', -5, 10)
    space.register(dim2)

    return space


@pytest.fixture
def tpe(space):
    """Return an instance of TPE."""
    return TPE(space, seed=1)


def test_compute_max_ei_point():
    """Test that max ei point is computed correctly"""
    points = numpy.linspace(-3, 3, num=10)
    below_likelis = numpy.linspace(0.5, 0.9, num=10)
    above_likes = numpy.linspace(0.2, 0.5, num=10)

    numpy.random.shuffle(below_likelis)
    numpy.random.shuffle(above_likes)
    max_ei_index = (below_likelis - above_likes).argmax()

    max_ei_point = compute_max_ei_point(points, below_likelis, above_likes)
    assert max_ei_point == points[max_ei_index]


def test_adaptive_parzen_normal_estimator():
    """Test adaptive parzen estimator"""
    low = -1
    high = 5

    obs_mus = [1.2]
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)
    assert list(mus) == [1.2, 2]
    assert list(sigmas) == [3, 6]
    assert list(weights) == [1.0 / 2, 1.0 / 2]

    obs_mus = [3.4]
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=0.5,
                                                     equal_weight=False, flat_num=25)
    assert list(mus) == [2, 3.4]
    assert list(sigmas) == [6, 3]
    assert list(weights) == [0.5 / 1.5, 1.0 / 1.5]

    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = (numpy.concatenate([ramp, full]))

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[:30 - 25] == ramp / all_weights.sum())
    assert numpy.all(weights[30 - 25:] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_weight():
    """Test the weight for the normal components"""
    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    low = -1
    high = 5

    # equal weight
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=True, flat_num=25)
    assert numpy.all(weights == 1 / 31)
    assert numpy.all(sigmas == 6 / 10)

    # prior weight
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=0.5,
                                                     equal_weight=False, flat_num=25)

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = (numpy.concatenate([ramp, full]))
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 0.5

    assert numpy.all(weights[:30 - 25] == (numpy.linspace(1.0 / 30, 1.0, num=30 - 25) /
                                           all_weights.sum()))
    assert numpy.all(weights[33 - 25:prior_pos] == 1 / all_weights.sum())
    assert weights[prior_pos] == 0.5 / all_weights.sum()
    assert numpy.all(weights[prior_pos + 1:] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)

    # full weights number
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=15)

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 15)
    full = numpy.ones(15 + 1)
    all_weights = (numpy.concatenate([ramp, full]))
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 1.0

    assert numpy.all(weights[:30 - 15] == (numpy.linspace(1.0 / 30, 1.0, num=30 - 15) /
                                           all_weights.sum()))
    assert numpy.all(weights[30 - 15:] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_sigma_clip():
    """Test that the magic clip of sigmas for parzen estimator"""
    low = -1
    high = 5

    obs_mus = numpy.linspace(-1, 5, num=8, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)
    assert len(mus) == len(sigmas) == len(weights) == 8 + 1
    assert numpy.all(weights == 1 / 9)
    assert numpy.all(sigmas == 6 / 8)

    obs_mus = numpy.random.uniform(-1, 5, 30)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 10)

    obs_mus = numpy.random.uniform(-1, 5, 400)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)

    assert len(mus) == len(sigmas) == len(weights) == 400 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 20)

    obs_mus = numpy.random.uniform(-1, 5, 10000)
    mus, sigmas, weights = adaptive_parzen_estimator(obs_mus, low, high, prior_weight=1.0,
                                                     equal_weight=False, flat_num=25)

    assert len(mus) == len(sigmas) == len(weights) == 10000 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 100)


class TestGMMSampler():
    """Tests for TPE GMM Sampler"""

    def test_gmm_sampler_creation(self, tpe):
        """Test GMMSampler creation"""
        mus = numpy.linspace(-3, 3, num=12, endpoint=False)
        sigmas = [0.5] * 12

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -3, 3)

        assert len(gmm_sampler.weights) == 12
        assert len(gmm_sampler.pdfs) == 12

    def test_sample(self, tpe):
        """Test GMMSampler sample function"""
        mus = numpy.linspace(-3, 3, num=12, endpoint=False)
        sigmas = [0.5] * 12

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -3, 3)
        points = gmm_sampler.sample(25)
        points = numpy.array(points)

        assert len(points) <= 25
        assert numpy.all(points >= -3)
        assert numpy.all(points < 3)

        mus = numpy.linspace(-10, 10, num=10, endpoint=False)
        sigmas = [0.00001] * 10
        weights = numpy.linspace(1, 10, num=10) ** 3
        numpy.random.shuffle(weights)
        weights = weights / weights.sum()

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)
        points = gmm_sampler.sample(10000)
        points = numpy.array(points)
        hist = numpy.histogram(points, bins=[-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9])

        assert numpy.all(hist[0].argsort() == numpy.array(weights).argsort())
        assert numpy.all(points >= -11)
        assert numpy.all(points < 9)

    def test_get_loglikelis(self):
        """Test to get log likelis of points"""
        mus = numpy.linspace(-10, 10, num=10, endpoint=False)
        weights = numpy.linspace(1, 10, num=10) ** 3
        numpy.random.shuffle(weights)
        weights = weights / weights.sum()

        sigmas = [0.00001] * 10
        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)

        points = [mus[7]]
        pdf = norm(mus[7], sigmas[7])
        point_likeli = numpy.log(pdf.pdf(mus[7]) * weights[7])
        likelis = gmm_sampler.get_loglikelis(points)

        assert list(likelis) == point_likeli
        assert likelis[0] == point_likeli

        sigmas = [2] * 10
        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)

        log_pdf = []
        pdfs = []
        for i in range(10):
            pdfs.append(norm(mus[i], sigmas[i]))
        for pdf, weight in zip(pdfs, weights):
            log_pdf.append(numpy.log(pdf.pdf(0) * weight))
        point_likeli = numpy.log(numpy.sum(numpy.exp(log_pdf)))

        points = numpy.random.uniform(-11, 9, 30)
        points = numpy.insert(points, 10, 0)
        likelis = gmm_sampler.get_loglikelis(points)

        point_likeli = numpy.format_float_scientific(point_likeli, precision=10)
        gmm_likeli = numpy.format_float_scientific(likelis[10], precision=10)
        assert point_likeli == gmm_likeli
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

    def test_split_trials(self, tpe):
        """Test observed trials can be split based on TPE gamma"""
        space = Space()
        dim1 = Real('yolo1', 'uniform', -3, 6)
        space.register(dim1)

        tpe.space = space

        points = numpy.linspace(-3, 3, num=10, endpoint=False)
        results = numpy.linspace(0, 1, num=10, endpoint=False)
        points_results = list(zip(points, results))
        numpy.random.shuffle(points_results)
        points, results = zip(*points_results)
        for point, result in zip(points, results):
            tpe.observe([[point]], [{'objective': result}])

        tpe.gamma = 0.25
        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4], [-1.8]]
        assert len(above_points) == 7

        tpe.gamma = 0.2
        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4]]
        assert len(above_points) == 8

    def test_sample_real_dimension(self):
        """Test sample values for a real dimension"""
        space = Space()
        dim1 = Real('yolo1', 'uniform', -10, 20)
        space.register(dim1)
        dim2 = Real('yolo2', 'uniform', -5, 10, shape=(2))
        space.register(dim2)

        tpe = TPE(space)
        points = numpy.random.uniform(-10, 10, 20).reshape(20, 1)
        below_points = points[:6, :]
        above_points = points[6:, :]
        points = tpe.sample_real_dimension(dim1, 1, below_points, above_points)
        assert len(points) == 1

        points = numpy.random.uniform(-5, 5, 32).reshape(16, 2)
        below_points = points[:4, :]
        above_points = points[4:, :]
        points = tpe.sample_real_dimension(dim2, 2, below_points, above_points)
        assert len(points) == 2

        tpe.n_ei_candidates = 0
        points = tpe.sample_real_dimension(dim2, 2, below_points, above_points)
        assert len(points) == 0

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

    def test_suggest_initial_points(self, tpe, monkeypatch):
        """Test that initial points can be sampled correctly"""
        points = [(i, i**2) for i in range(1, 12)]

        global index
        index = 0

        def sample(num=1, seed=None):
            global index
            pts = points[index:index + num]
            index += num
            return pts

        monkeypatch.setattr(tpe.space, 'sample', sample)

        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(1, 11):
            point = tpe.suggest(1)[0]
            assert point == (i, i**2)
            tpe.observe([point], [{'objective': results[i - 1]}])

        point = tpe.suggest(1)[0]
        assert point != (11, 11 * 2)

    def test_suggest_ei_candidates(self, tpe):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 2
        tpe.n_ei_candidates = 0

        results = numpy.random.random(2)
        for i in range(2):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 2
            assert not isinstance(point[0][0], tuple)
            tpe.observe(point, [{'objective': results[i]}])

        point = tpe.suggest(1)
        assert not point

        tpe.n_ei_candidates = 24
        point = tpe.suggest(1)
        assert len(point) > 0
