# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.hyperband.hyperband -- TODO
=================================================

.. module:: hyperband
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import copy
import hashlib
import logging

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity


logger = logging.getLogger(__name__)


REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
Hyperband cannot be used if space does contain a fidelity dimension.
For more information on the configuration and usage of Hyperband, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#hyperband
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(max_resources, reduction_factor):
    """Compute the budgets used for each execution of hyperband"""
    num_brackets = int(numpy.log(max_resources) / numpy.log(reduction_factor))
    capital_b = (num_brackets + 1) * max_resources
    budgets = []
    for bracket_id in range(0, num_brackets + 1):
        bracket_budgets = []
        num_trials = capital_b / max_resources * reduction_factor ** (num_brackets - bracket_id)
        min_resources = max_resources / reduction_factor ** (num_brackets - bracket_id)
        for i in range(0, num_brackets - bracket_id + 1):
            n_i = int(num_trials / reduction_factor ** i)
            min_i = int(numpy.ceil(min_resources * reduction_factor ** i))
            bracket_budgets.append((n_i, min_i))

        budgets.append(bracket_budgets)

    return budgets


class Hyperband(BaseAlgorithm):
    """Hyperband

    `Hyperparameter optimization [formulated] as a pure-exploration non-stochastic
    infinite-armed bandit problem where a predefined resource like iterations, data samples,
    or features is allocated to randomly sampled configurations.``

    For more information on the algorithm,
    see original paper at http://jmlr.org/papers/v18/16-558.html.

    Li, Lisha et al. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    Journal of Machine Learning Research, 18:1-52, 2018.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    """

    def __init__(self, space, seed=None, repeat=numpy.inf):
        self.brackets = []
        super(Hyperband, self).__init__(space, seed=seed)

        self.trial_info = {}  # Stores Trial -> Bracket

        try:
            fidelity_index = self.fidelity_index
        except IndexError:
            raise RuntimeError(SPACE_ERROR)

        fidelity_dim = space.values()[fidelity_index]

        # min_resources = fidelity_dim.low
        self.max_resources = fidelity_dim.high
        self.reduction_factor = fidelity_dim.base

        if self.reduction_factor < 2:
            raise AttributeError("Reduction factor for Hyperband needs to be at least 2.")

        self.repeat = repeat
        self.execution_times = 0

        self.budgets = compute_budgets(self.max_resources, self.reduction_factor)

        self.brackets = [
            Bracket(self, bracket_budgets)
            for bracket_budgets in self.budgets
        ]

        self.seed_rng(seed)

    def sample(self, num, bracket, buffer=10):
        """Sample new points from bracket"""
        samples = self.space.sample(num * buffer, seed=bracket.seed)
        i = 0
        points = []
        while len(points) < num and i < num * buffer:
            point = samples[i]
            if self.get_id(point) not in self.trial_info:
                point = list(point)
                point[self.fidelity_index] = bracket.rungs[0]['resources']
                points.append(tuple(point))
            i += 1

        if not points:
            raise RuntimeError(
                'Hyperband keeps sampling already existing points. This should not happen, '
                'please report this error to '
                'https://github.com/bouthilx/orion.algo.hyperband/issues')

        return points

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.seed = seed
        for i, bracket in enumerate(self.brackets):
            bracket.seed_rng(seed + i if seed is not None else None)
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {'rng_state': self.rng.get_state(), 'seed': self.seed}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.seed_rng(state_dict['seed'])
        self.rng.set_state(state_dict['rng_state'])

    def suggest(self, num=1):
        """Suggest a number of new sets of parameters.

        Sample new points until first rung is filled. Afterwards
        waits for all trials to be completed before promoting trials
        to the next rung.

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        """
        # TODO: now we are not suggesting the request number of points and far more than 1.
        if num > 1:
            raise ValueError("Hyperband should suggest only one point.")

        samples = []
        for bracket in reversed(self.brackets):
            if not bracket.is_filled:
                samples += bracket.sample()

        if samples:
            return samples

        # All brackets are filled

        for bracket in reversed(self.brackets):
            if bracket.is_ready() and not bracket.is_done:
                samples += bracket.promote()

        if samples:
            return samples

        # Either all brackets are done or none are ready and algo needs to wait for some trials to
        # complete
        return None

    def get_id(self, point):
        """Compute a unique hash for a point based on params, but not fidelity level."""
        _point = list(point)
        non_fidelity_dims = _point[0:self.fidelity_index]
        non_fidelity_dims.extend(_point[self.fidelity_index + 1:])

        return hashlib.md5(str(non_fidelity_dims).encode('utf-8')).hexdigest()

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        for point, result in zip(points, results):

            _id = self.get_id(point)
            bracket = self.trial_info.get(_id)

            if not bracket:
                fidelity = point[self.fidelity_index]
                brackets = [bracket for bracket in self.brackets
                            if bracket.rungs[0]['resources'] == fidelity]
                if not brackets:
                    raise ValueError(
                        "No bracket found for point {0} with fidelity {1}".format(_id, fidelity))
                bracket = brackets[0]

            try:
                bracket.register(point, result['objective'])
            except IndexError:
                logger.warning('Point registered to wrong bracket. This is likely due '
                               'to a corrupted database, where trials of different fidelity '
                               'have a wrong timestamps.')
                continue

            if _id not in self.trial_info:
                self.trial_info[_id] = bracket

        if all(bracket.is_done for bracket in self.brackets):
            self.execution_times += 1
            logger.debug('hyperband execution %i is done, required to execute %s times',
                         self.execution_times, str(self.repeat))

            # Continue to the next execution if need
            if self.execution_times < self.repeat:
                self.brackets = [
                    Bracket(self, bracket_budgets)
                    for bracket_budgets in self.budgets
                ]
                if self.seed is not None:
                    self.seed += 1

    @property
    def is_done(self):
        """Return True, if all required execution been done."""
        if self.execution_times == self.repeat:
            return True
        return False

    @property
    def fidelity_index(self):
        """Compute the index of the point when fidelity is."""
        def _is_fidelity(dim):
            return (isinstance(dim, Fidelity) or
                    (hasattr(dim, 'original_dimension') and
                     isinstance(dim.original_dimension, Fidelity)))

        return [i for i, dim in enumerate(self.space.values()) if _is_fidelity(dim)][0]


class Bracket():
    """Bracket of rungs for the algorithm Hyperband.

    Parameters
    ----------
    hyperband: `Hyperband` algorithm
        The hyperband algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung

    """

    def __init__(self, hyperband, budgets):
        self.hyperband = hyperband
        self.rungs = [dict(resources=budget, n_trials=n_trials, results=dict())
                      for n_trials, budget in budgets]

        self.seed = None

        logger.debug('Bracket budgets: %s', str(budgets))

        # points = hyperband.sample(compute_rung_sizes(reduction_factor, len(budgets))[0])
        # for point in points:
        #     self.register(point, None)

    @property
    def is_filled(self):
        """Return True if last rung with trials is filled"""
        return self.has_rung_filled(0)

    def sample(self):
        """Sample a new trial with lowest fidelity"""
        should_have_n_trials = self.rungs[0]['n_trials']
        return self.hyperband.sample(should_have_n_trials, self)

    def register(self, point, objective):
        """Register a point in the corresponding rung"""
        fidelity = point[self.hyperband.fidelity_index]
        rungs = [rung['results'] for rung in self.rungs if rung['resources'] == fidelity]
        if not rungs:
            budgets = [rung['resources'] for rung in self.rungs]
            raise IndexError(REGISTRATION_ERROR.format(fidelity=fidelity, budgets=budgets,
                                                       params=point))

        rungs[0][self.hyperband.get_id(point)] = (objective, point)

    def get_candidates(self, rung_id):
        """Get a candidate for promotion"""
        if self.has_rung_filled(rung_id + 1):
            return []

        rung = self.rungs[rung_id]['results']
        next_rung = self.rungs[rung_id + 1]['results']

        rung = list(sorted((objective, point) for objective, point in rung.values()))

        should_have_n_trials = self.rungs[rung_id + 1]['n_trials']
        points = []
        i = 0
        while len(points) + len(next_rung) < should_have_n_trials:
            objective, point = rung[i]
            assert objective is not None
            _id = self.hyperband.get_id(point)
            if _id not in next_rung:
                points.append(point)
            i += 1

        return points

    @property
    def is_done(self):
        """Return True, if the last rung is filled."""
        return self.has_rung_filled(len(self.rungs) - 1)

    def has_rung_filled(self, rung_id):
        """Return True, if the rung[rung_id] is filled."""
        n_trials = len(self.rungs[rung_id]['results'])
        return n_trials >= self.rungs[rung_id]['n_trials']

    def is_ready(self, rung_id=None):
        """Return True, if the bracket is ready for next promote"""
        if rung_id is not None:
            return (
                self.has_rung_filled(rung_id) and
                all(objective is not None
                    for objective, _ in self.rungs[rung_id]['results'].values()))

        is_ready = False
        for _rung_id in range(len(self.rungs)):
            if self.has_rung_filled(_rung_id):
                is_ready = self.is_ready(_rung_id)
            else:
                break

        return is_ready

    def promote(self):
        """Promote the first candidate that is found and return it

        The rungs are iterated over in reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted point.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        if self.is_done:
            return None

        for rung_id in range(len(self.rungs)):
            if self.has_rung_filled(rung_id + 1):
                continue

            if not self.is_ready(rung_id):
                return None

            points = []
            for candidate in self.get_candidates(rung_id):
                # pylint: disable=logging-format-interpolation
                logger.debug(
                    'Promoting {point} from rung {past_rung} with fidelity {past_fidelity} to '
                    'rung {new_rung} with fidelity {new_fidelity}'.format(
                        point=candidate, past_rung=rung_id,
                        past_fidelity=candidate[self.hyperband.fidelity_index],
                        new_rung=rung_id + 1, new_fidelity=self.rungs[rung_id + 1]['resources']))

                candidate = list(copy.deepcopy(candidate))
                candidate[self.hyperband.fidelity_index] = self.rungs[rung_id + 1]['resources']
                points.append(tuple(candidate))

            return points

        return None

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.seed = seed

    def __repr__(self):
        """Return representation of bracket with fidelity levels"""
        return 'Bracket({})'.format([rung['resources'] for rung in self.rungs])
