"""
MOFA Sampler module
"""

# from skopt.sampler import Lhs
# from skopt.space import Space as SkSpace

import logging

import numpy as np
from scipy.stats.qmc import LatinHypercube

logger = logging.getLogger(__name__)


def _is_prime(number):
    if number < 2 or (number % 2 == 0 and number > 2):
        return False
    for i in range(3, int(np.sqrt(number - 1)) + 1, 2):
        if number % i == 0:
            return False
    return True


def _find_n_levels(dimensions):
    levels = dimensions - 1
    while True:
        if _is_prime(levels):
            return levels
        else:
            levels = levels + 1


def generate_olh_samples(space, n_levels, strength, index, rng):
    """
    Generates samples from an orthogonal Latin hypercube (OLH)

    Parameters
    ----------
    space: `orion.core.worker.Space`
        Parameter space
    n_levels: int
        Number of levels
    strength: {1,2}
        Strength parameter for an orthogonal Latin hypercube
    rng: None or ``numpy.random.RandomState``
        Random number generator

    Returns
    -------
    (numpy.array, int) A tuple of the samples array from the normalized parameter space,
        and the n_levels parameter, which may be changed to suit the OLH requirements.
    """

    dimensions = len(space.items())
    if not _is_prime(n_levels):
        n_levels = _find_n_levels(dimensions)
        logger.warning(
            """WARNING: n_levels specified is not a prime number.
            Changing n_levels to %d""",
            n_levels,
        )
    elif n_levels < dimensions - 1:
        n_levels = _find_n_levels(dimensions)
        logger.warning(
            """WARNING: n_levels specified is less than the number of hyperparameters
            minus 1. Changing n_levels to %d""",
            n_levels,
        )
    n_rows = n_levels**strength
    logger.debug(
        "MOFA: setting number of trials in this iteration to %d", n_rows * index
    )
    all_samples = []
    for _ in range(index):
        lhc = LatinHypercube(d=dimensions, strength=strength, seed=rng)
        samples = lhc.random(n_rows).tolist()
        all_samples.extend(samples)

    return np.array(all_samples), n_levels


def generate_trials(olh_samples, roi_space):
    """
    Generates trials from the given normalized orthogonal Latin hypercube samples

    Parameters
    ----------
    olh_samples: `numpy.array`
        Samples from the orthogonal Latin hypercube
    roi_space: orion.algo.space.Space
        Parameter space region-of-interest

    Returns
    -------
    A list of trials as `dict` objects, each a list of parameter values in the
        original search space
    """

    trials = []
    for sample in olh_samples:
        trial_dict = {}
        for j, param_name in enumerate(roi_space.keys()):
            interval_min, interval_max = roi_space[param_name].interval()
            # TODO: deal with categoricals
            trial_dict[param_name] = (
                sample[j] * (interval_max - interval_min) + interval_min
            )
        trials.append(trial_dict)
    return trials
