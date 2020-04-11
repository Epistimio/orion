# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.tpe` -- Tree-structured Parzen Estimator Approach
==================================================================

.. module:: tpe
   :platform: Unix
   :synopsis: Tree-structured Parzen Estimator Approach

"""

import numpy
from scipy.stats import norm

from orion.algo.base import BaseAlgorithm
from orion.core.utils.points import flatten_dims, regroup_dims


def compute_max_ei_point(points, below_likelis, above_likelis):
    """Compute ei among points based on their log likelihood and return the point with max ei.

    :param points: list of point with real values.
    :param below_likelis: list of log likelihood for each point in the good GMM.
    :param above_likelis: list of log likelihood for each point in the bad GMM.
    """
    max_ei = -numpy.inf
    point_index = 0
    for i, (lik_b, lik_a) in enumerate(zip(below_likelis, above_likelis)):
        ei = lik_b - lik_a
        if ei > max_ei:
            max_ei = ei
            point_index = i
    return points[point_index]


# pylint:disable=assignment-from-no-return
def adaptive_parzen_estimator(mus, low, high,
                              prior_weight=1.0,
                              equal_weight=False,
                              flat_num=25):
    """Return the sorted mus, the corresponding sigmas and weights with adaptive kernel estimator.

    :param mus: list of real values for observed mus.
    :param low: real value for lower bound of points.
    :param high: real value for upper bound of points.
    :param prior_weight: real value for the weight of prior point.
    :param equal_weight: bool value indicating if all points with equal weights.
    :param flat_num: int value indicating the number of latest points with equal weights,
                     it is only valid if `equal_weight` is False.
    """
    def update_weights(total_num):
        """Generate weights for all components"""
        if total_num < flat_num or equal_weight:
            return numpy.ones(total_num)

        ramp_weights = numpy.linspace(1.0 / total_num, 1.0, num=total_num - flat_num)
        flat_weights = numpy.ones(flat_num)
        return numpy.concatenate([ramp_weights, flat_weights])

    mus = numpy.asarray(mus)

    prior_mu = (low + high) * 0.5
    prior_sigma = (high - low) * 1.0

    size = len(mus)
    if size > 1:
        order = numpy.argsort(mus)
        sorted_mus = mus[order]
        prior_mu_pos = numpy.searchsorted(sorted_mus, prior_mu)

        weights = update_weights(size)

        mixture_mus = numpy.zeros(size + 1)
        mixture_mus[:prior_mu_pos] = sorted_mus[:prior_mu_pos]
        mixture_mus[prior_mu_pos] = prior_mu
        mixture_mus[prior_mu_pos + 1:] = sorted_mus[prior_mu_pos:]

        mixture_weights = numpy.ones(size + 1)
        mixture_weights[:prior_mu_pos] = weights[:prior_mu_pos]
        mixture_weights[prior_mu_pos] = prior_weight
        mixture_weights[prior_mu_pos + 1:] = weights[prior_mu_pos:]

        sigmas = numpy.ones(size + 1)
        sigmas[0] = mixture_mus[1] - mixture_mus[0]
        sigmas[-1] = mixture_mus[-1] - mixture_mus[-2]
        sigmas[1:-1] = numpy.maximum((mixture_mus[1:-1] - mixture_mus[0:-2]),
                                     (mixture_mus[2:] - mixture_mus[1:-1]))
        sigmas = numpy.clip(sigmas, prior_sigma / max(10, numpy.sqrt(size)), prior_sigma)

    else:
        if prior_mu < mus[0]:
            mixture_mus = numpy.array([prior_mu, mus[0]])
            sigmas = numpy.array([prior_sigma, prior_sigma * 0.5])
            mixture_weights = numpy.array([prior_weight, 1.0])
        else:
            mixture_mus = numpy.array([mus[0], prior_mu])
            sigmas = numpy.array([prior_sigma * 0.5, prior_sigma])
            mixture_weights = numpy.array([1.0, prior_weight])

    weights = mixture_weights / len(mixture_weights)

    return mixture_mus, sigmas, weights


class TPE(BaseAlgorithm):
    """Tree-structured Parzen Estimator (TPE) algorithm is one of Sequential Model-Based
    Global Optimization (SMBO) algorithms, which will build models to propose new points based
    on the historical observed trials.

    Instead of modeling p(y|x) like other SMBO algorithms, TPE models p(x|y) and p(y),
    and p(x|y) is modeled by transforming that generative process, replacing the distributions of
    the configuration prior with non-parametric densities.

    The TPE defines p(xjy) using two such densities l(x) and g(x) while l(x) is distribution of
    good points and g(x) is the distribution of bad points. New point candidates will be sampled
    with l(x) and Expected Improvement (EI) optimization scheme will be used to find the most
    promising point among the candidates.

    For more information on the algorithm, see original paper at
    https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed to sample initial points and candidates points.
        Default: ``None``
    n_initial_points: int
        Number of initial points randomly sampled.
        Default: ``20``
    n_ei_candidates: int
        Number of candidates points sampled for ei compute.
        Default: ``24``
    gamma: real
        Ratio to split the observed trials into good and bad distributions.
        Default: ``0.25``
    equal_weight: bool
        True to set equal weights for observed points.
        Default: ``False``
    prior_weight: int
        The weight given to the prior point of the input space.
        Default: ``1.0``

    """

    def __init__(self, space, seed=None,
                 n_initial_points=20, n_ei_candidates=24,
                 gamma=0.25, equal_weight=False, prior_weight=1.0):

        super(TPE, self).__init__(space,
                                  seed=seed,
                                  n_initial_points=n_initial_points,
                                  n_ei_candidates=n_ei_candidates,
                                  gamma=gamma,
                                  equal_weight=equal_weight,
                                  prior_weight=prior_weight)

        for _, dimension in self.space.items():

            if dimension.type not in ['real']:
                raise ValueError("TPE now only supports Real Dimension.")

            if dimension.prior_name not in ['uniform']:
                raise ValueError("TPE now only supports uniform as prior.")

            shape = dimension.shape
            if shape and len(shape) != 1:
                raise ValueError("TPE now only supports 1D shape.")

        self.seed_rng(seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict = super(TPE, self).state_dict

        _state_dict['rng_state'] = self.rng.get_state()
        _state_dict['seed'] = self.seed
        return _state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(TPE, self).set_state(state_dict)

        self.seed_rng(state_dict['seed'])
        self.rng.set_state(state_dict['rng_state'])

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        if num > 1:
            raise ValueError("TPE should suggest only one point.")

        samples = []
        if len(self._trials_info) < self.n_initial_points:
            new_point = self.space.sample(1, seed=tuple(self.rng.randint(0, 1000000, size=3)))[0]
            print(type(new_point), type(new_point[0]), new_point)
            samples.append(new_point)
        else:
            point = []
            below_points, above_points = self.split_trials()
            below_points = numpy.array([flatten_dims(point, self.space) for point in below_points])
            above_points = numpy.array([flatten_dims(point, self.space) for point in above_points])
            idx = 0
            for i, dimension in enumerate(self.space.values()):

                if dimension.type == 'real':
                    shape = dimension.shape
                    if not shape:
                        shape = (1,)
                    # Unpack dimension
                    for j in range(shape[0]):
                        idx = idx + j
                        new_point = self._sample_real(dimension,
                                                      below_points[:, idx],
                                                      above_points[:, idx])
                        point.append(new_point)

                else:
                    raise ValueError("TPE now only support Real Dimension.")

                idx += 1
            point = regroup_dims(point, self.space)
            samples.append(point)
        return samples

    def _sample_real(self, dimension, below_points, above_points):
        """Return a real point based on the observed good and bad points"""
        low, high = dimension.interval()
        below_mus, below_sigmas, blow_weights = \
            adaptive_parzen_estimator(below_points, low, high, self.prior_weight,
                                      self.equal_weight, flat_num=25)
        above_mus, above_sigmas, above_weights = \
            adaptive_parzen_estimator(above_points, low, high, self.prior_weight,
                                      self.equal_weight, flat_num=25)

        gmm_sampler_below = GMMSampler(self, below_mus, below_sigmas, low, high, blow_weights)
        gmm_sampler_above = GMMSampler(self, above_mus, above_sigmas, low, high, above_weights)

        points = gmm_sampler_below.sample(self.n_ei_candidates)
        lik_blow = gmm_sampler_below.get_loglikelis(points)
        lik_above = gmm_sampler_above.get_loglikelis(points)

        new_point = compute_max_ei_point(points, lik_blow, lik_above)

        return new_point

    def split_trials(self):
        """Split the observed trials into good and bad ones based on the ratio `gamma``"""
        sorted_trials = sorted(self._trials_info.values(), key=lambda x: x[1]['objective'])
        sorted_points = [list(points) for points, results in sorted_trials]

        split_index = int(numpy.ceil(self.gamma * len(sorted_points)))

        below = sorted_points[:split_index]
        above = sorted_points[split_index:]
        print(below)

        return below, above

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        super(TPE, self).observe(points, results)


class GMMSampler():
    """Gaussian Mixture Model Sampler for TPE algorithm

    Parameters
    ----------
    tpe: `TPE` algorithm
        The tpe algorithm object which this sampler will be part of.
    mus: list
        mus for each Gaussian components in the GMM.
        Default: ``None``
    sigmas: list
        sigmas for each Gaussian components in the GMM.
    low: real
        Lower bound of the sampled points.
    high: real
        Upper bound of the sampled points.
    weights: list
        Weights for each Gaussian components in the GMM
        Default: ``None``

    """

    def __init__(self, tpe, mus, sigmas, low, high, weights=None):
        self.tpe = tpe

        self.mus = mus
        self.sigmas = sigmas
        self.low = low
        self.high = high
        self.weights = weights if weights is not None else len(mus) * [1.0 / len(mus)]

        self.pdfs = []
        self._build_mixture()

    def _build_mixture(self):
        """Build the Gaussian components in the GMM"""
        for mu, sigma in zip(self.mus, self.sigmas):
            self.pdfs.append(norm(mu, sigma))

    def sample(self, num=1):
        """Sample required number of points"""
        point = []
        for _ in range(num):
            pdf = numpy.argmax(self.tpe.rng.multinomial(1, self.weights))
            new_points = self.pdfs[pdf].rvs(size=5)
            for pt in new_points:
                if self.low <= pt < self.high:
                    point.append(pt)
                    break

        return point

    def get_loglikelis(self, points):
        """Return the log likelihood for the points"""
        points = numpy.array(points)
        weight_likelis = [numpy.log(self.weights[i] * pdf.pdf(points))
                          for i, pdf in enumerate(self.pdfs)]
        weight_likelis = numpy.array(weight_likelis)
        weight_likelis = weight_likelis.transpose()

        # log-sum-exp trick
        max_likeli = numpy.nanmax(weight_likelis, axis=1)
        point_likeli = max_likeli + numpy.log(numpy.nansum
                                              (numpy.exp(weight_likelis - max_likeli[:, None]),
                                               axis=1))

        return point_likeli
