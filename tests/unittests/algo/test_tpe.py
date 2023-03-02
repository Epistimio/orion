#!/usr/bin/env python
"""Tests for :mod:`orion.algo.tpe`."""
from __future__ import annotations

import copy
import itertools
import timeit
from typing import ClassVar

import numpy
import numpy as np
import pytest
from scipy.stats import norm

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space
from orion.algo.tpe import (
    TPE,
    CategoricalSampler,
    GMMSampler,
    adaptive_parzen_estimator,
    compute_max_ei_point,
    ramp_up_weights,
)
from orion.core.utils import backward, format_trials
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.transformer import build_required_space
from orion.core.worker.trial import Trial
from orion.testing.algo import BaseAlgoTests, TestPhase, first_phase_only


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()

    dim1 = Real("yolo1", "uniform", -10, 20)
    space.register(dim1)

    dim2 = Integer("yolo2", "uniform", -5, 10)
    space.register(dim2)

    categories = ["a", 0.1, 2, "c"]
    dim3 = Categorical("yolo3", categories)
    space.register(dim3)

    return space


@pytest.fixture
def tpe(space: Space):
    """Return an instance of TPE."""
    return TPE(space, seed=1)


def test_compute_max_ei_point():
    """Test that max ei point is computed correctly"""
    points = numpy.linspace(-3, 3, num=10).tolist()
    below_likelis = numpy.linspace(0.5, 0.9, num=10)
    above_likes = numpy.linspace(0.2, 0.5, num=10)

    numpy.random.shuffle(below_likelis)
    numpy.random.shuffle(above_likes)
    max_ei_index = (below_likelis - above_likes).argmax()  # type: ignore

    max_ei_point = compute_max_ei_point(points, below_likelis, above_likes)
    assert max_ei_point == points[max_ei_index]


def test_ramp_up_weights():
    """Test TPE adjust observed points correctly"""
    weights = ramp_up_weights(25, 15, True)
    assert len(weights) == 25
    assert numpy.all(weights == 1.0)

    weights = ramp_up_weights(25, 15, False)
    assert len(weights) == 25
    assert numpy.all(weights[:10] == (numpy.linspace(1.0 / 25, 1.0, num=10)))
    assert numpy.all(weights[10:] == 1.0)

    weights = ramp_up_weights(10, 15, False)
    assert len(weights) == 10
    assert numpy.all(weights == 1.0)

    weights = ramp_up_weights(25, 0, False)
    assert len(weights) == 25
    assert numpy.all(weights == (numpy.linspace(1.0 / 25, 1.0, num=25)))


def test_adaptive_parzen_normal_estimator():
    """Test adaptive parzen estimator"""
    low = -1
    high = 5

    obs_mus = [1.2]
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )
    assert list(mus) == [1.2, 2]
    assert list(sigmas) == [3, 6]
    assert list(weights) == [1.0 / 2, 1.0 / 2]

    obs_mus = [3.4]
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=0.5, equal_weight=False, flat_num=25
    )
    assert list(mus) == [2, 3.4]
    assert list(sigmas) == [6, 3]
    assert list(weights) == [0.5 / 1.5, 1.0 / 1.5]

    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = numpy.concatenate([ramp, full])

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[: 30 - 25] == ramp / all_weights.sum())
    assert numpy.all(weights[30 - 25 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_weight():
    """Test the weight for the normal components"""
    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    low = -1
    high = 5

    # equal weight
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=True, flat_num=25
    )
    assert numpy.all(weights == 1 / 31)
    assert numpy.all(sigmas == 6 / 10)

    # prior weight
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=0.5, equal_weight=False, flat_num=25
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = numpy.concatenate([ramp, full])
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 0.5

    assert numpy.all(
        weights[: 30 - 25]
        == (numpy.linspace(1.0 / 30, 1.0, num=30 - 25) / all_weights.sum())
    )
    assert numpy.all(weights[33 - 25 : prior_pos] == 1 / all_weights.sum())
    assert weights[prior_pos] == 0.5 / all_weights.sum()
    assert numpy.all(weights[prior_pos + 1 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)

    # full weights number
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=15
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 15)
    full = numpy.ones(15 + 1)
    all_weights = numpy.concatenate([ramp, full])
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 1.0

    assert numpy.all(
        weights[: 30 - 15]
        == (numpy.linspace(1.0 / 30, 1.0, num=30 - 15) / all_weights.sum())
    )
    assert numpy.all(weights[30 - 15 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_sigma_clip():
    """Test that the magic clip of sigmas for parzen estimator"""
    low = -1
    high = 5

    obs_mus = numpy.linspace(-1, 5, num=8, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )
    assert len(mus) == len(sigmas) == len(weights) == 8 + 1
    assert numpy.all(weights == 1 / 9)
    assert numpy.all(sigmas == 6 / 8)

    obs_mus = numpy.random.uniform(-1, 5, 30)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 10)

    obs_mus = numpy.random.uniform(-1, 5, 400)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 400 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 20)

    obs_mus = numpy.random.uniform(-1, 5, 10000)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 10000 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 100)


class TestCategoricalSampler:
    """Tests for TPE Categorical Sampler"""

    def test_cat_sampler_creation(self, tpe: TPE):
        """Test CategoricalSampler creation"""
        obs: list | numpy.ndarray = [0, 3, 9]
        choices = list(range(-5, 5))
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        assert len(cat_sampler.weights) == len(choices)

        obs = numpy.array([0, 3, 9])
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        assert len(cat_sampler.weights) == len(choices)

        tpe.equal_weight = True
        tpe.prior_weight = 1.0
        obs = numpy.random.randint(0, 10, 100)
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        counts_obs = numpy.bincount(obs) + 1.0
        weights = counts_obs / counts_obs.sum()

        assert numpy.all(cat_sampler.weights == weights)

        tpe.equal_weight = False
        tpe.prior_weight = 0.5
        tpe.full_weight_num = 30
        obs = numpy.random.randint(0, 10, 100)

        cat_sampler = CategoricalSampler(tpe, obs, choices)

        ramp = numpy.linspace(1.0 / 100, 1.0, num=100 - 30)
        full = numpy.ones(30)
        ramp_weights = numpy.concatenate([ramp, full])

        counts_obs = numpy.bincount(obs, weights=ramp_weights) + 0.5
        weights = counts_obs / counts_obs.sum()

        assert numpy.all(cat_sampler.weights == weights)

    @pytest.mark.parametrize("seed", [123, 456])
    def test_sample(self, tpe: TPE, seed: int):
        """Test CategoricalSampler sample function"""
        rng = numpy.random.RandomState(seed)
        obs = rng.randint(0, 10, 100)
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        points = cat_sampler.sample(25)

        assert len(points) == 25
        assert numpy.all(points >= 0)
        assert numpy.all(points < 10)

        weights = numpy.linspace(1, 10, num=10) ** 3
        rng.shuffle(weights)
        weights = weights / weights.sum()
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        cat_sampler.weights = weights

        points = cat_sampler.sample(10000)
        points = numpy.array(points)
        hist = numpy.bincount(points)

        assert numpy.all(hist.argsort() == weights.argsort())
        assert len(points) == 10000
        assert numpy.all(points >= 0)
        assert numpy.all(points < 10)

    def test_get_loglikelis(self, tpe: TPE):
        """Test to get log likelis of points"""
        obs = numpy.random.randint(0, 10, 100)
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        points = cat_sampler.sample(25)

        likelis = cat_sampler.get_loglikelis(points)

        assert numpy.all(
            likelis == numpy.log(numpy.asarray(cat_sampler.weights)[points])
        )


class TestGMMSampler:
    """Tests for TPE GMM Sampler"""

    def test_gmm_sampler_creation(self, tpe: TPE):
        """Test GMMSampler creation"""
        mus = numpy.linspace(-3, 3, num=12, endpoint=False)
        sigmas = [0.5] * 12

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -3, 3)

        assert len(gmm_sampler.weights) == 12
        assert len(gmm_sampler.pdfs) == 12

    def test_sample(self, tpe: TPE):
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

    def test_sample_narrow_space(self, tpe: TPE):
        """Test that sampling in a narrow space does not fail to fast"""
        mus = numpy.ones(12) * 0.5
        sigmas = [0.5] * 12

        times = []
        for bounds in [(0.4, 0.6), (0.49, 0.51), (0.499, 0.501), (0.49999, 0.50001)]:
            gmm_sampler = GMMSampler(tpe, mus, sigmas, *bounds)
            times.append(timeit.timeit(lambda: gmm_sampler.sample(2), number=100))

        # Test that easy sampling takes less time.
        assert sorted(times) == times

        gmm_sampler = GMMSampler(
            tpe, mus, sigmas, 0.05, 0.04, attempts_factor=1, max_attempts=10
        )
        with pytest.raises(RuntimeError) as exc:
            gmm_sampler.sample(1, attempts=10)

        assert exc.match("Failed to sample in interval")

    def test_get_loglikelis(self, tpe: TPE):
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


def _trial_to_array(trial: Trial, space: Space) -> list[tuple]:
    return list(format_trials.trial_to_tuple(trial, space=space))


def _array_to_trial(x: np.ndarray, space: Space, y: np.ndarray | None = None) -> Trial:
    trial = format_trials.tuple_to_trial(x, space=space)
    if y is not None:
        trial.results = [_result(y)]
        trial.status = "completed"
    return trial


def _result(y: float | np.ndarray) -> Trial.Result:
    return Trial.Result(name="objective", type="objective", value=float(y))


def _add_result(trial: Trial, y: float) -> Trial:
    trial = copy.deepcopy(trial)
    trial.results = [_result(y)]
    trial.status = "completed"
    return trial


class TestTPE_old:
    """Tests for the algo TPE."""

    def test_unsupported_space(self):
        """Test tpe only work for supported search space"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -10, 10)
        space.register(dim1)
        dim2 = Real("yolo2", "reciprocal", 10, 20)
        space.register(dim2)
        categories = ["a", 0.1, 2, "c"]
        dim3 = Categorical("yolo3", categories)
        space.register(dim3)
        dim4 = Fidelity("epoch", 1, 9, 3)
        space.register(dim4)
        TPE(space)

        space = Space()
        dim = Real("yolo1", "norm", 0.9)
        space.register(dim)

        with pytest.raises(
            ValueError,
            match="TPE now only supports uniform, loguniform, uniform discrete and choices",
        ):
            tpe = TPE(
                space=build_required_space(space, shape_requirement=TPE.requires_shape)
            )

    def test_split_trials(self):
        """Test observed trials can be split based on TPE gamma"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -3, 6)
        space.register(dim1)
        tpe = TPE(space, seed=1)
        rng = np.random.RandomState(1)
        points = numpy.linspace(-3, 3, num=10, endpoint=False).reshape(-1, 1)
        objectives = numpy.linspace(0, 1, num=10, endpoint=False)
        point_objectives = list(zip(points, objectives))
        rng.shuffle(point_objectives)
        points, objectives = zip(*point_objectives)
        for point, objective in zip(points, objectives):
            trial = _array_to_trial(point, space=tpe.space, y=objective)
            tpe.observe([trial])

        tpe.gamma = 0.25
        below_trials, above_trials = tpe.split_trials()
        below_points = [_trial_to_array(t, space=tpe.space) for t in below_trials]
        assert below_points == [[-3.0], [-2.4], [-1.8]]
        assert len(above_trials) == 7

        tpe.gamma = 0.2
        below_trials, above_trials = tpe.split_trials()
        below_points = [_trial_to_array(t, space=tpe.space) for t in below_trials]
        assert below_points == [[-3.0], [-2.4]]
        assert len(above_trials) == 8

    @pytest.mark.xfail(reason="TODO: Need to update this test, TPE changed a bit.")
    def test_sample_int_dimension(self):
        """Test sample values for a integer dimension"""
        space = Space()
        dim1 = Integer("yolo1", "uniform", -10, 20)
        space.register(dim1)

        dim2 = Integer("yolo2", "uniform", -5, 10, shape=(2))
        space.register(dim2)

        tpe = TPE(space)

        obs_points = numpy.random.randint(-10, 10, 100)
        below_points = [obs_points[:25]]
        above_points = [obs_points[25:]]
        # BUG: This should FAIL!
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 10)

        obs_points_below = numpy.random.randint(-10, 0, 25).reshape(1, 25)
        obs_points_above = numpy.random.randint(0, 10, 75).reshape(1, 75)
        points = tpe.sample_one_dimension(
            dim1, 1, obs_points_below, obs_points_above, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 0)

        obs_points = numpy.random.randint(-5, 5, 100)
        below_points = [obs_points[:25], obs_points[25:50]]
        above_points = [obs_points[50:75], obs_points[75:]]
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 2
        assert all(points >= -10)
        assert all(points < 10)

        tpe.n_ei_candidates = 0
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_int_point
        )
        assert len(points) == 0

    @pytest.mark.xfail(reason="TODO: Need to update this test, TPE changed a bit.")
    def test_sample_categorical_dimension(self):
        """Test sample values for a categorical dimension"""
        space = Space()
        categories = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        dim1 = Categorical("yolo1", categories)
        space.register(dim1)
        dim2 = Categorical("yolo2", categories, shape=(2))
        space.register(dim2)

        tpe = TPE(space)

        obs_points = numpy.random.randint(0, 10, 100)
        obs_points = [categories[point] for point in obs_points]
        below_points = [obs_points[:25]]
        above_points = [obs_points[25:]]
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 1
        assert points[0] in categories

        obs_points_below = numpy.random.randint(0, 3, 25)
        obs_points_above = numpy.random.randint(3, 10, 75)
        below_points = [[categories[point] for point in obs_points_below]]
        above_points = [[categories[point] for point in obs_points_above]]
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 1
        assert points[0] in categories[:3]

        obs_points = numpy.random.randint(0, 10, 100)
        obs_points = [categories[point] for point in obs_points]
        below_points = [obs_points[:25], obs_points[25:50]]
        above_points = [obs_points[50:75], obs_points[75:]]

        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 2
        assert points[0] in categories
        assert points[1] in categories

        tpe.n_ei_candidates = 0
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 0

    @pytest.mark.xfail(reason="TODO: Need to update this test, TPE changed a bit.")
    def test_sample_real_dimension(self):
        """Test sample values for a real dimension"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -10, 20)
        space.register(dim1)
        dim2 = Real("yolo2", "uniform", -5, 10, shape=(2))
        space.register(dim2)
        dim3 = Real("yolo3", "reciprocal", 1, 20)
        space.register(dim3)

        tpe = TPE(space)
        points = numpy.random.uniform(-10, 10, 20)
        below_points = [points[:8]]
        above_points = [points[8:]]
        points = tpe._sample_real_dimension(dim1, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 10)

        points = numpy.random.uniform(1, 20, 20)
        below_points = [points[:8]]
        above_points = [points[8:]]
        points = tpe._sample_real_dimension(dim3, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= 1)
        assert all(points < 20)

        below_points = numpy.random.uniform(-10, 0, 25).reshape(1, 25)
        above_points = numpy.random.uniform(0, 10, 75).reshape(1, 75)
        points = tpe._sample_real_dimension(dim1, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 0)

        points = numpy.random.uniform(-5, 5, 32)
        below_points = [points[:8], points[8:16]]
        above_points = [points[16:24], points[24:]]
        points = tpe._sample_real_dimension(dim2, 2, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 2
        assert all(points >= -10)
        assert all(points < 10)

        tpe.n_ei_candidates = 0
        points = tpe._sample_real_dimension(dim2, 2, below_points, above_points)
        assert len(points) == 0

    def test_suggest(self, tpe: TPE):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(10):
            trials = tpe.suggest(1)
            assert trials is not None
            assert len(trials) == 1
            points = [_trial_to_array(t, space=tpe.space) for t in trials]
            assert len(points[0]) == 3
            assert not isinstance(points[0][0], tuple)
            trials[0] = _add_result(trials[0], results[i])
            tpe.observe(trials)

        trials = tpe.suggest(1)
        assert trials is not None
        assert len(trials) == 1
        points = [_trial_to_array(t, space=tpe.space) for t in trials]
        assert len(points[0]) == 3
        assert not isinstance(points[0][0], tuple)

    def test_1d_shape(self):
        """Test suggest with 1D shape dimensions"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -3, 6, shape=(2))
        space.register(dim1)
        dim2 = Real("yolo2", "uniform", -2, 4)
        space.register(dim2)

        tpe = create_algo(space=space, algo_type=TPE, seed=1, n_initial_points=10)
        results = numpy.random.random(10)
        for i in range(10):
            trials = tpe.suggest(1)
            assert trials is not None
            assert len(trials) == 1
            points = [_trial_to_array(t, space=tpe.space) for t in trials]
            assert len(points[0]) == 2
            assert len(points[0][0]) == 2
            trials[0] = _add_result(trials[0], results[i])
            tpe.observe(trials)

        trials = tpe.suggest(1)
        assert trials is not None
        assert len(trials) == 1
        points = [_trial_to_array(t, space=tpe.space) for t in trials]
        assert len(points[0]) == 2
        assert len(points[0][0]) == 2

    @pytest.mark.xfail(
        reason="TODO: Adapt this test if it's relevant. Wasn't run previously."
    )
    def test_suggest_initial_points(self, tpe: TPE, monkeypatch):
        """Test that initial points can be sampled correctly"""
        _points = [(i, i - 6, "c") for i in range(1, 12)]
        _trials = [
            format_trials.tuple_to_trial(point, space=tpe.space) for point in _points
        ]
        index = 0

        def sample(num: int = 1, seed=None) -> list[Trial]:
            nonlocal index
            result = _trials[index : index + num]
            index += num
            return result

        monkeypatch.setattr(tpe.space, "sample", sample)

        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(1, 11):
            trials = tpe.suggest(1)
            assert trials is not None
            trial = trials[0]
            assert trial.params == _trials[i]
            point = format_trials.trial_to_tuple(trial, space=tpe.space)
            assert point == (i, i - 6, "c")
            trial.results = [
                Trial.Result(name="objective", type="objective", value=results[i - 1])
            ]
            tpe.observe([trial])

        trials = tpe.suggest(1)
        assert trials is not None
        trial = trials[0]
        assert trial == _trials[-1]
        # BUG: This is failing. We expect this trial to be sampled from the model, not from the
        # search space.
        assert format_trials.trial_to_tuple(trial, space=tpe.space) != (11, 5, "c")

    @pytest.mark.xfail(
        reason="TODO: Adapt/debug Test if useful. (wasn't run at all before, is failing now)."
    )
    def test_suggest_ei_candidates(self, tpe: TPE):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 2
        tpe.n_ei_candidates = 0

        results = numpy.random.random(2)
        for i in range(2):
            trials = tpe.suggest(1)
            assert trials is not None
            assert len(trials) == 1
            points = [format_trials.trial_to_tuple(trials[0], space=tpe.space)]
            assert len(points[0]) == 3
            assert not isinstance(points[0][0], tuple)
            trials[0] = _add_result(trials[0], results[i])
            tpe.observe(trials)

        trials = tpe.suggest(1)
        assert not trials

        tpe.n_ei_candidates = 24
        trials = tpe.suggest(1)
        assert trials is not None
        assert len(trials) > 0


N_INIT = 10


class TestTPE(BaseAlgoTests):
    algo_name = "tpe"
    config = {
        "seed": 123456,
        "n_initial_points": N_INIT,
        "n_ei_candidates": 4,
        "gamma": 0.5,
        "equal_weight": True,
        "prior_weight": 0.8,
        "full_weight_num": 10,
        "max_retry": 100,
        "parallel_strategy": {
            "of_type": "StatusBasedParallelStrategy",
            "strategy_configs": {
                "broken": {"of_type": "MaxParallelStrategy", "default_result": 100},
            },
            "default_strategy": {
                "of_type": "meanparallelstrategy",
                "default_result": 50,
            },
        },
    }
    max_trials: ClassVar[int] = 30

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("bo", N_INIT, "_suggest_bo"),
    ]

    @first_phase_only
    def test_suggest_init(self, first_phase: TestPhase):
        """Test that the first call to `suggest` returns all the random initial points
        (and only those).
        """
        algo = self.create_algo()
        # Ask for more than the number of points in the first phase.
        points = algo.suggest(first_phase.length * 3)
        assert points is not None
        assert len(points) == first_phase.length

    @first_phase_only
    def test_suggest_init_missing(self, first_phase: TestPhase):
        algo = self.create_algo()
        missing = 3
        self.force_observe(algo=algo, num=first_phase.length - missing)
        # Ask for more than the number of points in the first phase.
        points = algo.suggest(first_phase.length * 3)
        assert points is not None
        assert len(points) == missing

    @first_phase_only
    def test_suggest_init_overflow(self, mocker, first_phase: TestPhase):
        algo = self.create_algo()

        self.force_observe(algo=algo, num=first_phase.length - 1)
        spy = mocker.spy(algo.algorithm.space, "sample")
        # Now reaching end of the first phase, by asking more trials than the length of the
        # first phase.
        trials = algo.suggest(first_phase.length * 3)
        assert trials is not None
        assert len(trials) == 1

        # Verify trial was sampled randomly, not using BO
        assert spy.call_count == 1

        # Next call to suggest should still be in first phase, since we still haven't observed that
        # missing random trial.
        trials = algo.suggest(first_phase.length * 3)
        assert trials is not None
        assert len(trials) == 1
        # Verify trial was sampled randomly, not using BO
        assert spy.call_count == 2

    def test_suggest_n(self):
        """Verify that suggest returns correct number of trials if ``num`` is specified in ``suggest``."""
        algo = self.create_algo()
        trials = algo.suggest(5)
        assert trials is not None
        # TODO: Why did this change? Why did it use to return 5 trials?
        if self._current_phase.name == "bo":
            assert len(trials) == 1
        else:
            assert len(trials) == 5

    @first_phase_only
    def test_thin_real_space(self, monkeypatch, first_phase: TestPhase):
        algo = self.create_algo()

        self.force_observe(num=first_phase.length, algo=algo)

        original_sample = GMMSampler.sample

        low = 0.5
        high = 0.5000001

        def sample(self, num, attempts=None):
            self.attempts_factor = 1
            self.max_attempts = 10
            self.low = low
            self.high = high
            return original_sample(self, num, attempts=attempts)

        monkeypatch.setattr(GMMSampler, "sample", sample)
        with pytest.raises(
            RuntimeError, match=rf"Failed to sample in interval \({low}, {high}\)"
        ):
            algo.suggest(1)

    @first_phase_only
    def test_is_done_cardinality(self):
        # TODO: Support correctly loguniform(discrete=True)
        #       See https://github.com/Epistimio/orion/issues/566
        space_dict = {
            "x": "uniform(0, 4, discrete=True)",
            "y": "choices(['a', 'b', 'c'])",
            "z": "uniform(1, 6, discrete=True)",
        }
        space = self.create_space(space_dict)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        # Prevent the algo from exiting early because of a max_trials limit.
        algo.algorithm.max_trials = None
        i = 0
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            backward.algo_observe(
                algo,
                [format_trials.tuple_to_trial([x, y, z], space)],
                [dict(objective=i)],
            )
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done

    @first_phase_only
    def test_log_integer(self, monkeypatch):
        """Verify that log integer dimensions do not go out of bound."""
        RANGE = 100
        # NOTE: Here we're passing the 'original' space, not the transformed space.
        algo = self.create_algo(
            space=self.create_space({"x": f"loguniform(1, {RANGE}, discrete=True)"}),
        )

        algo.algorithm.max_trials = RANGE * 2
        values = list(range(1, RANGE + 1))

        # Mock sampling so that it quickly samples all possible integers in given bounds
        def sample(self, n_samples=1, seed=None):
            return [
                format_trials.tuple_to_trial(
                    (numpy.log(values.pop()),), algo.algorithm.space
                )
                for _ in range(n_samples)
            ]

        def _suggest_random(self, num: int) -> list[Trial]:
            v = self._suggest(num, sample)
            print(f"num={num}, Suggesting {v} ({len(values)} values left.)")
            return v

        monkeypatch.setattr(TPE, "_suggest_random", _suggest_random)
        monkeypatch.setattr(TPE, "_suggest_bo", _suggest_random)

        self.force_observe(RANGE, algo)
        transform_wrapper = algo.unwrap(SpaceTransform)
        assert len(algo.algorithm.registry) == RANGE
        assert len(transform_wrapper.registry_mapping) == RANGE
        assert len(transform_wrapper.registry) == RANGE
        assert algo.algorithm.n_suggested == RANGE
        assert algo.algorithm.n_observed == RANGE
        assert transform_wrapper.n_suggested == RANGE
        assert transform_wrapper.n_observed == RANGE

    @first_phase_only
    def test_stuck_exploiting(self, monkeypatch):
        """Test that algo drops out when exploiting an already explored region."""
        algo = self.create_algo()

        trials = algo.space.sample(1)
        assert trials is not None

        # Mock sampling so that always returns the same trials
        def sample(self, n_samples: int = 1, seed=None) -> list[Trial]:
            return trials

        def _suggest_random(self: TPE, num: int) -> list[Trial]:
            return self._suggest(num, sample)

        monkeypatch.setattr(TPE, "_suggest_random", _suggest_random)
        monkeypatch.setattr(TPE, "_suggest_bo", _suggest_random)

        with pytest.raises(RuntimeError):
            self.force_observe(2, algo)

        assert algo.n_observed == 1
        assert algo.n_suggested == 1
