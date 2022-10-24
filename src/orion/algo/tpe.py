"""
Tree-structured Parzen Estimator Approach
=========================================
"""
from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar, Iterable, Sequence, TypeVar

import numpy
from scipy.stats import norm
from scipy.stats.distributions import rv_frozen

from orion.algo.base import BaseAlgorithm
from orion.algo.parallel_strategy import ParallelStrategy, strategy_factory
from orion.algo.space import Integer, Real, Space
from orion.core.utils import format_trials
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)
T = TypeVar("T")


def compute_max_ei_point(
    points: numpy.ndarray | Sequence[numpy.ndarray],
    below_likelis: Sequence[float] | numpy.ndarray,
    above_likelis: Sequence[float] | numpy.ndarray,
) -> numpy.ndarray:
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


def ramp_up_weights(total_num: int, flat_num: int, equal_weight: bool) -> numpy.ndarray:
    """Adjust weights of observed trials.

    :param total_num: total number of observed trials.
    :param flat_num: the number of the most recent trials which
        get the full weight where the others will be applied with a linear ramp
        from 0 to 1.0. It will only take effect if equal_weight is False.
    :param equal_weight: whether all the observed trails share the same weights.
    """
    if total_num < flat_num or equal_weight:
        return numpy.ones(total_num)

    ramp_weights = numpy.linspace(1.0 / total_num, 1.0, num=total_num - flat_num)
    flat_weights = numpy.ones(flat_num)
    return numpy.concatenate([ramp_weights, flat_weights])


# pylint:disable=assignment-from-no-return
def adaptive_parzen_estimator(
    mus: numpy.ndarray | Sequence,
    low: float,
    high: float,
    prior_weight: float = 1.0,
    equal_weight: bool = False,
    flat_num: int = 25,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Return the sorted mus, the corresponding sigmas and weights with adaptive kernel estimator.

    This adaptive parzen window estimator is based on the original papers and also refer the use of
    prior mean in `this implementation
    <https://github.com/hyperopt/hyperopt/blob/40bd617a0ca47e09368fb65919464ba0bfa85962/hyperopt/tpe.py#L400-L468>`_.

    :param mus: list of real values for observed mus.
    :param low: real value for lower bound of points.
    :param high: real value for upper bound of points.
    :param prior_weight: real value for the weight of the prior mean.
    :param equal_weight: bool value indicating if all points with equal weights.
    :param flat_num: int value indicating the number of the most recent trials which
        get the full weight where the others will be applied with a linear ramp
        from 0 to 1.0. It will only take effect if equal_weight is False.
    """
    mus = numpy.asarray(mus)

    prior_mu = (low + high) * 0.5
    prior_sigma = (high - low) * 1.0

    size = len(mus)
    if size > 1:
        order = numpy.argsort(mus)
        sorted_mus = mus[order]
        prior_mu_pos = numpy.searchsorted(sorted_mus, prior_mu)

        weights = ramp_up_weights(size, flat_num, equal_weight)

        mixture_mus = numpy.zeros(size + 1)
        mixture_mus[:prior_mu_pos] = sorted_mus[:prior_mu_pos]
        mixture_mus[prior_mu_pos] = prior_mu
        mixture_mus[prior_mu_pos + 1 :] = sorted_mus[prior_mu_pos:]

        mixture_weights = numpy.ones(size + 1)
        mixture_weights[:prior_mu_pos] = weights[:prior_mu_pos]
        mixture_weights[prior_mu_pos] = prior_weight
        mixture_weights[prior_mu_pos + 1 :] = weights[prior_mu_pos:]

        sigmas = numpy.ones(size + 1)
        sigmas[0] = mixture_mus[1] - mixture_mus[0]
        sigmas[-1] = mixture_mus[-1] - mixture_mus[-2]
        sigmas[1:-1] = numpy.maximum(
            (mixture_mus[1:-1] - mixture_mus[0:-2]),
            (mixture_mus[2:] - mixture_mus[1:-1]),
        )
        sigmas = numpy.clip(
            sigmas, prior_sigma / max(10, numpy.sqrt(size)), prior_sigma
        )

    else:
        if prior_mu < mus[0]:

            mixture_mus = numpy.array([prior_mu, mus[0]])
            sigmas = numpy.array([prior_sigma, prior_sigma * 0.5])
            mixture_weights = numpy.array([prior_weight, 1.0])
        else:
            mixture_mus = numpy.array([mus[0], prior_mu])
            sigmas = numpy.array([prior_sigma * 0.5, prior_sigma])
            mixture_weights = numpy.array([1.0, prior_weight])

    weights = mixture_weights / mixture_weights.sum()

    return mixture_mus, sigmas, weights


class TPE(BaseAlgorithm):
    """Tree-structured Parzen Estimator (TPE) algorithm is one of Sequential Model-Based
    Global Optimization (SMBO) algorithms, which will build models to propose new points based
    on the historical observed trials.

    Instead of modeling p(y|x) like other SMBO algorithms, TPE models p(x|y) and p(y),
    and p(x|y) is modeled by transforming that generative process, replacing the distributions of
    the configuration prior with non-parametric densities.

    The TPE defines p(x|y) using two such densities l(x) and g(x) while l(x) is distribution of
    good points and g(x) is the distribution of bad points. New point candidates will be sampled
    with l(x) and Expected Improvement (EI) optimization scheme will be used to find the most
    promising point among the candidates.

    For more information on the algorithm, see original papers at:

    - `Algorithms for Hyper-Parameter Optimization
      <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
    - `Making a Science of Model Search: Hyperparameter Optimizationin Hundreds of Dimensions
      for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int, optional
        Seed to sample initial points and candidates points.
        Default: ``None``
    n_initial_points: int, optional
        Number of initial points randomly sampled. If new points
        are requested and less than `n_initial_points` are observed,
        the next points will also be sampled randomly instead of being
        sampled from the parzen estimators.
        Default: ``20``
    n_ei_candidates: int, optional
        Number of candidates points sampled for ei compute. Larger numbers will lead to more
        exploitation and lower numbers will lead to more exploration. Be careful with categorical
        dimension as TPE tend to severily exploit these if n_ei_candidates is larger than 1.
        Default: ``24``
    gamma: real, optional
        Ratio to split the observed trials into good and bad distributions. Lower numbers will
        load to more exploitation and larger numbers will lead to more exploration.
        Default: ``0.25``
    equal_weight: bool, optional
        True to set equal weights for observed points.
        Default: ``False``
    prior_weight: int, optional
        The weight given to the prior point of the input space.
        Default: ``1.0``
    full_weight_num: int, optional
        The number of the most recent trials which get the full weight where the others will be
        applied with a linear ramp from 0 to 1.0. It will only take effect if equal_weight
        is False.
    max_retry: int, optional
        Number of attempts to sample new points if the sampled points were already suggested.
        Default: ``100``
    parallel_strategy: dict or None, optional
        The configuration of a parallel strategy to use for pending trials or broken trials.
        Default is a MaxParallelStrategy for broken trials and NoParallelStrategy for pending
        trials.

    """

    requires_type: ClassVar[str | None] = None
    requires_dist: ClassVar[str | None] = "linear"
    requires_shape: ClassVar[str | None] = "flattened"

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = None,
        n_initial_points: int = 20,
        n_ei_candidates: int = 24,
        gamma: float = 0.25,
        equal_weight: bool = False,
        prior_weight: float = 1.0,
        full_weight_num: int = 25,
        max_retry: int = 100,
        parallel_strategy: dict | None = None,
    ):

        if n_initial_points < 2:
            n_initial_points = 2
            logger.warning(
                "n_initial_points %s is not valid, set n_initial_points = 2",
                str(n_initial_points),
            )

        if n_ei_candidates < 1:
            n_ei_candidates = 1
            logger.warning(
                "n_ei_candidates %s is not valid, set n_ei_candidates = 1",
                str(n_ei_candidates),
            )

        if parallel_strategy is None:
            parallel_strategy = {
                "of_type": "StatusBasedParallelStrategy",
                "strategy_configs": {
                    "broken": {
                        "of_type": "MaxParallelStrategy",
                    },
                },
            }

        self.strategy: ParallelStrategy = strategy_factory.create(**parallel_strategy)

        super().__init__(
            space,
            seed=seed,
            n_initial_points=n_initial_points,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            equal_weight=equal_weight,
            prior_weight=prior_weight,
            full_weight_num=full_weight_num,
            max_retry=max_retry,
            parallel_strategy=parallel_strategy,
        )
        # TODO: Setting the attributes here so their types are registered. Need to get rid of
        # overlap with super().__init__ somehow.
        self.seed = seed
        self.n_initial_points = n_initial_points
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.equal_weight = equal_weight
        self.prior_weight = prior_weight
        self.full_weight_num = full_weight_num
        self.max_retry = max_retry
        self.parallel_strategy = parallel_strategy
        self._verify_space()

    def _verify_space(self) -> None:
        """Verify space is supported"""

        for dimension in self.space.values():

            if dimension.type != "fidelity" and dimension.prior_name not in [
                "uniform",
                "reciprocal",
                "int_uniform",
                "int_reciprocal",
                "choices",
            ]:
                raise ValueError(
                    "TPE now only supports uniform, loguniform, uniform discrete "
                    f"and choices as prior: {dimension.prior_name}"
                )

            shape = dimension.shape
            if shape and len(shape) != 1:
                raise ValueError("TPE now only supports 1D shape.")

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        super().seed_rng(seed)
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict: dict[str, Any] = super().state_dict
        _state_dict["rng_state"] = self.rng.get_state()
        _state_dict["seed"] = self.seed
        _state_dict["strategy"] = self.strategy.state_dict
        return _state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)

        self.seed_rng(state_dict["seed"])
        self.rng.set_state(state_dict["rng_state"])
        self.strategy.set_state(state_dict["strategy"])

    def suggest(self, num: int) -> list[Trial] | None:
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        Parameters
        ----------
        num: int
            Number of trials to sample.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        # Only sample up to `n_initial_points` and after that only sample one at a time.
        num_samples = min(num, max(self.n_initial_points - self.n_suggested, 1))

        samples: list[Trial] = []
        while len(samples) < num_samples and self.n_suggested < self.space.cardinality:
            if self.n_observed < self.n_initial_points:
                candidates = self._suggest_random(num_samples)
                logger.debug("Random candidates: %s", candidates)
            else:
                v = max(num_samples - len(samples), 0)
                candidates = self._suggest_bo(v)
                logger.debug("BO candidates: %s", candidates)

            if not candidates:
                # Perhaps the BO algo wasn't able to suggest any points? Break in that case.
                break
            for candidate in candidates:
                self.register(candidate)
                samples.append(candidate)
        return samples

    def _suggest(
        self, num: int, function: Callable[[int], Iterable[Trial]]
    ) -> list[Trial]:
        trials: list[Trial] = []

        retries = 0
        while len(trials) < num and retries < self.max_retry:
            for candidate in function(num - len(trials)):
                if candidate not in self.registry:
                    self.register(candidate)
                    trials.append(candidate)
                else:
                    retries += 1

                if len(self.registry) >= self.space.cardinality:
                    return trials

        if retries >= self.max_retry:
            logger.warning(
                f"Algorithm unable to sample `{num}` trials with less than "
                f"`{self.max_retry}` retries. Try adjusting the configuration of TPE "
                "to favor exploration (`n_ei_candidates` and `gamma` in particular)."
            )

        return trials

    def _suggest_random(self, num: int) -> list[Trial]:
        def sample(num: int) -> list[Trial]:
            return self.space.sample(
                num, seed=tuple(self.rng.randint(0, 1000000, size=3))
            )

        return self._suggest(num, sample)

    def _suggest_bo(self, num: int) -> list[Trial]:
        def suggest_bo(num: int) -> list[Trial]:
            return [self._suggest_one_bo() for _ in range(num)]

        return self._suggest(num, suggest_bo)

    def _suggest_one_bo(self) -> Trial:

        params = {}
        below_trials, above_trials = self.split_trials()

        for dimension in self.space.values():
            dim_below_trials = [trial.params[dimension.name] for trial in below_trials]
            dim_above_trials = [trial.params[dimension.name] for trial in above_trials]

            if dimension.type == "real":
                dim_samples = self._sample_real_dimension(
                    dimension,
                    dim_below_trials,
                    dim_above_trials,
                )
            elif dimension.type == "integer" and dimension.prior_name in [
                "int_uniform",
                "int_reciprocal",
            ]:
                dim_samples = self._sample_int_point(
                    dimension,
                    dim_below_trials,
                    dim_above_trials,
                )
            elif dimension.type == "categorical" and dimension.prior_name == "choices":
                dim_samples = self._sample_categorical_point(
                    dimension,
                    dim_below_trials,
                    dim_above_trials,
                )
            elif dimension.type == "fidelity":
                # fidelity dimension
                trial = self.space.sample(1)[0]
                dim_samples = trial.params[dimension.name]
            else:
                raise NotImplementedError()

            params[dimension.name] = dim_samples

        trial = format_trials.dict_to_trial(params, self.space)
        return trial

    def _sample_real_dimension(
        self,
        dimension: Real,
        below_points: Sequence[numpy.ndarray],
        above_points: Sequence[numpy.ndarray],
    ) -> numpy.ndarray:
        """Sample values for real dimension"""
        if any(map(dimension.prior_name.endswith, ["uniform", "reciprocal"])):
            return self._sample_real_point(
                dimension,
                below_points,
                above_points,
            )
        else:
            raise NotImplementedError(
                f"Prior {dimension.prior_name} is not supported for real values"
            )

    def _sample_loguniform_real_point(
        self, dimension: Real, below_points: numpy.ndarray, above_points: numpy.ndarray
    ) -> numpy.ndarray:
        """Sample one value for real dimension in a loguniform way"""
        return self._sample_real_point(
            dimension, below_points, above_points, is_log=True
        )

    def _sample_real_point(
        self,
        dimension: Real,
        below_points: numpy.ndarray | Sequence[numpy.ndarray],
        above_points: numpy.ndarray | Sequence[numpy.ndarray],
        is_log: bool = False,
    ) -> numpy.ndarray:
        """Sample one value for real dimension based on the observed good and bad points"""
        low, high = dimension.interval()
        below_points = numpy.array(below_points)
        above_points = numpy.array(above_points)
        if is_log:
            low = numpy.log(low)
            high = numpy.log(high)
            below_points = numpy.log(below_points)
            above_points = numpy.log(above_points)

        below_mus, below_sigmas, below_weights = adaptive_parzen_estimator(
            below_points,
            low,
            high,
            self.prior_weight,
            self.equal_weight,
            flat_num=self.full_weight_num,
        )
        above_mus, above_sigmas, above_weights = adaptive_parzen_estimator(
            above_points,
            low,
            high,
            self.prior_weight,
            self.equal_weight,
            flat_num=self.full_weight_num,
        )

        gmm_sampler_below = GMMSampler(
            self,
            mus=below_mus,
            sigmas=below_sigmas,
            low=low,
            high=high,
            weights=below_weights,
        )
        gmm_sampler_above = GMMSampler(
            self,
            mus=above_mus,
            sigmas=above_sigmas,
            low=low,
            high=high,
            weights=above_weights,
        )

        candidate_points = gmm_sampler_below.sample(self.n_ei_candidates)
        lik_blow = gmm_sampler_below.get_loglikelis(candidate_points)
        lik_above = gmm_sampler_above.get_loglikelis(candidate_points)
        new_point = compute_max_ei_point(candidate_points, lik_blow, lik_above)

        if is_log:
            new_point = numpy.exp(new_point)

        return new_point

    def _sample_int_point(
        self,
        dimension: Integer,
        below_points: numpy.ndarray | Sequence[numpy.ndarray],
        above_points: numpy.ndarray | Sequence[numpy.ndarray],
    ):
        """Sample one value for integer dimension based on the observed good and bad points"""
        low, high = dimension.interval()
        assert isinstance(low, int)
        assert isinstance(high, int)
        choices = range(low, high + 1)

        below_points = numpy.array(below_points).astype(int) - low
        above_points = numpy.array(above_points).astype(int) - low

        sampler_below = CategoricalSampler(self, below_points, choices)
        candidate_points = sampler_below.sample(self.n_ei_candidates)

        sampler_above = CategoricalSampler(self, above_points, choices)

        lik_below = sampler_below.get_loglikelis(candidate_points)
        lik_above = sampler_above.get_loglikelis(candidate_points)

        new_point: numpy.ndarray = compute_max_ei_point(
            candidate_points, lik_below, lik_above
        )
        new_point = new_point + low
        return new_point

    def _sample_categorical_point(self, dimension, below_points, above_points):
        """Sample one value for categorical dimension based on the observed good and bad points"""
        choices = dimension.interval()

        below_points = [choices.index(point) for point in below_points]
        above_points = [choices.index(point) for point in above_points]

        sampler_below = CategoricalSampler(self, below_points, choices)
        candidate_points = sampler_below.sample(self.n_ei_candidates)

        sampler_above = CategoricalSampler(self, above_points, choices)

        lik_below = sampler_below.get_loglikelis(candidate_points)
        lik_above = sampler_above.get_loglikelis(candidate_points)

        new_point_index = compute_max_ei_point(candidate_points, lik_below, lik_above)
        new_point = choices[new_point_index]

        return new_point

    def split_trials(self) -> tuple[list[Trial], list[Trial]]:
        """Split the observed trials into good and bad ones based on the ratio `gamma``"""

        trials: list[Trial] = []
        for trial in self.registry:
            if trial.status != "completed":
                trial = self.strategy.infer(trial)

            if trial is not None:
                trials.append(trial)
        # NOTE: This assumes that all trials have an objective. Making assumption explicit.
        assert all(trial.objective is not None for trial in trials)
        sorted_trials = sorted(trials, key=lambda trial: trial.objective.value)  # type: ignore

        split_index = int(numpy.ceil(self.gamma * len(sorted_trials)))

        below = sorted_trials[:split_index]
        above = sorted_trials[split_index:]

        return below, above


class GMMSampler:
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
    base_attempts: int, optional
        Base number of attempts to sample points within `low` and `high` bounds.
        Defaults to 10.
    attempts_factor: int, optional
        If sampling always falls out of bound try again with `attempts` * `attempts_factor`.
        Defaults to 10.
    max_attempts: int, optional
        If sampling always falls out of bound try again with `attempts` * `attempts_factor`
        up to `max_attempts` (inclusive).
        Defaults to 10000.
    """

    def __init__(
        self,
        tpe: TPE,
        mus: Sequence[float] | numpy.ndarray,
        sigmas: Sequence[float] | numpy.ndarray,
        low: float,
        high: float,
        weights: Sequence[float] | numpy.ndarray | None = None,
        base_attempts: int = 10,
        attempts_factor: int = 10,
        max_attempts: int = 10000,
    ):
        self.tpe = tpe

        self.mus = mus
        self.sigmas = sigmas
        self.low = low
        self.high = high
        self.weights = weights if weights is not None else len(mus) * [1.0 / len(mus)]

        self.base_attempts = base_attempts
        self.attempts_factor = attempts_factor
        self.max_attempts = max_attempts

        self.pdfs: list[rv_frozen] = []
        self._build_mixture()

    def _build_mixture(self) -> None:
        """Build the Gaussian components in the GMM"""
        for mu, sigma in zip(self.mus, self.sigmas):
            self.pdfs.append(norm(mu, sigma))

    def sample(self, num: int = 1, attempts: int | None = None) -> list[numpy.ndarray]:
        """Sample required number of points"""
        if attempts is None:
            attempts = self.base_attempts
        point: list[numpy.ndarray] = []
        for _ in range(num):
            pdf = numpy.argmax(self.tpe.rng.multinomial(1, self.weights))
            attempts_tried = 0
            index: int | None = None
            while attempts_tried < attempts:
                new_points: numpy.ndarray = self.pdfs[pdf].rvs(
                    size=attempts, random_state=self.tpe.rng
                )
                valid_points = (self.low <= new_points) * (self.high >= new_points)

                if any(valid_points):
                    index = numpy.argmax(valid_points)
                    point.append(new_points[index])
                    break

                index = None
                attempts_tried += 1

            if index is None and attempts >= self.max_attempts:
                raise RuntimeError(
                    f"Failed to sample in interval ({self.low}, {self.high})"
                )
            elif index is None:
                point.append(
                    self.sample(num=1, attempts=attempts * self.attempts_factor)[0]
                )

        return point

    def get_loglikelis(
        self, points: numpy.ndarray | list[numpy.ndarray] | Sequence[Sequence[float]]
    ) -> numpy.ndarray:
        """Return the log likelihood for the points"""
        points = numpy.array(points)
        weight_likelis_list = [
            numpy.log(self.weights[i] * pdf.pdf(points))
            for i, pdf in enumerate(self.pdfs)
        ]
        weight_likelis = numpy.array(weight_likelis_list)
        # (num_weights, num_points) => (num_points, num_weights)
        weight_likelis = weight_likelis.transpose()

        # log-sum-exp trick
        max_likeli = numpy.nanmax(weight_likelis, axis=1)
        point_likeli = max_likeli + numpy.log(
            numpy.nansum(numpy.exp(weight_likelis - max_likeli[:, None]), axis=1)
        )

        return point_likeli


class CategoricalSampler:
    """Categorical Sampler for discrete integer and categorical choices

    Parameters
    ----------
    tpe: `TPE` algorithm
        The tpe algorithm object which this sampler will be part of.
    observations: list
        Observed values in the dimension
    choices: list
        Candidate values for the dimension

    """

    def __init__(
        self,
        tpe: TPE,
        observations: Sequence | numpy.ndarray,
        choices: Sequence | numpy.ndarray,
    ):
        self.tpe = tpe
        self.obs = observations
        self.choices = choices

        self._build_multinomial_weights()
        self.weights: numpy.ndarray

    def _build_multinomial_weights(self) -> None:
        """Build weights for categorical distribution based on observations"""
        weights_obs = ramp_up_weights(
            len(self.obs), self.tpe.full_weight_num, self.tpe.equal_weight
        )
        counts_obs = numpy.bincount(
            self.obs, minlength=len(self.choices), weights=weights_obs
        )
        counts_obs = counts_obs + self.tpe.prior_weight
        self.weights = counts_obs / counts_obs.sum()

    def sample(self, num: int = 1) -> numpy.ndarray:
        """Sample required number of points"""
        samples = self.tpe.rng.multinomial(n=1, pvals=self.weights, size=num)

        assert samples.shape == (num,) + (len(self.weights),)

        samples_index = samples.argmax(-1)
        assert samples_index.shape == (num,)

        return samples_index

    def get_loglikelis(
        self, points: numpy.ndarray | Sequence[numpy.ndarray]
    ) -> numpy.ndarray:
        """Return the log likelihood for the points"""
        return numpy.log(numpy.asarray(self.weights)[points])
