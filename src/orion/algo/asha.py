# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.asha` -- TODO
======================================================================

.. module:: asha
   :platform: Unix
   :synopsis: TODO

"""
import copy
import hashlib

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity


class ASHA(BaseAlgorithm):
    """Implement an algorithm that samples randomly from the problem's space."""

    def __init__(self, space, seed=None, max_resources=100, grace_period=1, reduction_factor=4,
                 num_brackets=1):
        """Random sampler takes no other hyperparameter than the problem's space
        itself.

        :param space: `orion.algo.space.Space` of optimization.
        :param seed: Integer seed for the random number generator.
        """
        super(ASHA, self).__init__(
            space, seed=seed, max_resources=max_resources, grace_period=grace_period,
            reduction_factor=reduction_factor, num_brackets=num_brackets)

        if reduction_factor < 2:
            raise AttributeError("Reduction factor for ASHA needs to be at least 2.")

        self.trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self.brackets = [
            Bracket(self, grace_period, max_resources, reduction_factor, s)
            for s in range(num_brackets)
        ]

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {'rng_state': self.rng.get_state()}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.seed_rng(0)
        self.rng.set_state(state_dict['rng_state'])

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        if num > 1:
            raise ValueError("ASHA should suggest only one point.")

        for bracket in self.brackets:
            candidate = bracket.update_rungs()

            if candidate:
                return [candidate]

        point = list(self.space.sample(num, seed=self.rng.randint(0, 10000))[0])

        sizes = numpy.array([len(b.rungs) for b in self.brackets])
        probs = numpy.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = self.rng.choice(len(self.brackets), p=normalized)

        point[self.fidelity_index] = self.brackets[idx].rungs[0][0]
        self.trial_info[self.get_id(point)] = self.brackets[idx]

        return [tuple(point)]

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
                            if bracket.rungs[0][0] == fidelity]
                if not brackets:
                    raise ValueError(
                        "No bracket found for point {0} with fidelity {1}".format(_id, fidelity))
                bracket = brackets[0]

            try:
                bracket.register(point, result['objective'])
            except IndexError as e:
                raise RuntimeError('Point registered to wrong bracket. This is likely due '
                                   'to a corrupted database, where trials of different fidelity '
                                   'have a wrong timestamps. Please report this to '
                                   'https://github.com/Epistimio/orion/issues') from e

            if _id not in self.trial_info:
                self.trial_info[_id] = bracket

    @property
    def is_done(self):
        """Return True, if all brackets reached their maximum resources."""
        return all(bracket.is_done for bracket in self.brackets)

    @property
    def fidelity_index(self):
        """Compute the index of the point when fidelity is."""
        def _is_fidelity(dim):
            return (isinstance(dim, Fidelity) or
                    (hasattr(dim, 'original_dimension') and
                     isinstance(dim.original_dimension, Fidelity)))

        return [i for i, dim in enumerate(self.space.values()) if _is_fidelity(dim)][0]


class Bracket():
    """Bracket of rungs for the algorithm ASHA."""

    def __init__(self, asha, min_t, max_t, reduction_factor, s):
        """Build rungs based on min_t, max_t, reduction_factor and s.

        :param asha: `ASHA` algorithm
        :param min_t: Minimum resources (grace_period)
        :param max_t: Maximum resources
        :param reduction_factor: Factor of reduction from `min_t` to `max_t`
        :param s: Minimal early stopping factor (used when there is many brackets)
        """
        if min_t <= 0:
            raise AttributeError("Minimum resources must be a positive number.")
        elif min_t > max_t:
            raise AttributeError("Minimum resources must be smaller than maximum resources.")

        self.asha = asha
        self.reduction_factor = reduction_factor
        max_rungs = int(numpy.log(max_t / min_t) / numpy.log(reduction_factor) - s + 1)
        self.rungs = [(min_t * reduction_factor**(k + s), dict())
                      for k in range(max_rungs)]

    def register(self, point, objective):
        """Register a point in the corresponding rung"""
        fidelity = point[self.asha.fidelity_index]
        rungs = [rung for budget, rung in self.rungs if budget == fidelity]
        if not rungs:
            budgets = [budget for budget, rung in self.rungs]
            raise IndexError('Bad fidelity level {}. Should be in {}'.format(fidelity, budgets))

        rungs[0][self.asha.get_id(point)] = (objective, point)

    def get_candidate(self, rung_id):
        """Get a candidate for promotion"""
        _, rung = self.rungs[rung_id]
        next_rung = self.rungs[rung_id + 1][1]

        rung = list(sorted((objective, point) for objective, point in rung.values()
                           if objective is not None))
        k = len(rung) // self.reduction_factor
        k = min(k, len(rung))

        for i in range(k):
            point = rung[i][1]
            _id = self.asha.get_id(point)
            if _id not in next_rung:
                return point

        return None

    @property
    def is_done(self):
        """Return True, if reached the bracket reached its maximum resources."""
        return len(self.rungs[-1][1])

    def update_rungs(self):
        """Promote the first candidate that is found and return it

        The rungs are iterated over is reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted point.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        # NOTE: There should be base + 1 rungs
        for rung_id in range(len(self.rungs) - 2, -1, -1):
            candidate = self.get_candidate(rung_id)
            if candidate:
                candidate = list(copy.deepcopy(candidate))
                candidate[self.asha.fidelity_index] = self.rungs[rung_id + 1][0]

                return tuple(candidate)

        return None
