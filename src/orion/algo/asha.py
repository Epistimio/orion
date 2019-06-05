# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.asha` -- TODO
======================================================================

.. module:: asha
   :platform: Unix
   :synopsis: TODO

"""
import hashlib

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity


class ASHA(BaseAlgorithm):
    """Implement an algorithm that samples randomly from the problem's space."""

    def __init__(self, space, seed=None, max_resources=100, grace_period=1, reduction_factor=4,
                 brackets=1):
        """Random sampler takes no other hyperparameter than the problem's space
        itself.

        :param space: `orion.algo.space.Space` of optimization.
        :param seed: Integer seed for the random number generator.
        """
        super(ASHA, self).__init__(
            space, seed=seed, max_resources=max_resources, grace_period=grace_period,
            reduction_factor=reduction_factor, brackets=brackets)

        if reduction_factor < 2:
            raise AttributeError("Reduction factor for ASHA needs to be at least 2.")

        self.trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self.brackets = [
            _Bracket(self, grace_period, max_resources, reduction_factor, s)
            for s in range(brackets)
        ]

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """

        for bracket in self.brackets:
            candidate = bracket.update_rungs()
            if candidate:
                return candidate

        point = self.space.sample(num, seed=self.rng.randint(0, 10000))

        sizes = numpy.array([len(b.rungs) for b in self.brackets])
        probs = numpy.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = numpy.random.choice(len(self.brackets), p=normalized)

        point[self.fidelity_index] = self.brackets[idx].rungs[-1][0]
        self.trial_info[self._get_id(point)] = self.brackets[idx]
        # NOTE: The point is not registered in bracket here, until in `observe()`

        return point

    def _get_id(self, point):
        non_fidelity_dims = point[0:self.fidelity_index]
        non_fidelity_dims.extend(point[self.fidelity_index + 1:])

        return hashlib.md5((non_fidelity_dims).encode('utf-8')).hexdigest()

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        for point, result in zip(points, results):

            _id = self._get_id(point)

            if _id not in self.trial_info:
                fidelity = point[self.fidelity_index]
                bracket = None

                for _bracket in self.brackets:
                    budget = _bracket.rungs[-1][0]

                    if fidelity == budget:
                        bracket = _bracket
                        break

                if bracket is None:
                    raise RuntimeError("No bracket found for point {0} with fidelity {1}".format(
                                       _id, fidelity)
                                       )
            else:
                bracket = self.trial_info[_id]

            bracket.register(point, result)

    def is_done(self):
        return all(bracket.is_done for bracket in self.brackets)

    @property
    def fidelity_index(self):
        return [i for i, dim in enumerate(self.space.values()) if isinstance(dim, Fidelity)][0]


class _Bracket():
    def __init__(self, asha, min_t, max_t, reduction_factor, s):
        if min_t <= 0:
            raise AttributeError("Minimum resources must be a positive number.")
        elif min_t > max_t:
            raise AttributeError("Minimum resources must be smaller than maximum resources.")

        self.asha = asha
        self.reduction_factor = reduction_factor
        max_rungs = int(numpy.log(max_t / min_t) / numpy.log(reduction_factor) - s + 1)
        self.rungs = [(min_t * reduction_factor**(k + s), dict())
                      for k in range(max_rungs + 1)]

    def register(self, point, objective):
        if point[self.asha.fidelity_index] != self.rungs[0][0]:
            raise AttributeError("Point {} budget different than rung 0.".format(
                                 self.asha._get_id(point)))

        self.rungs[0][1][self.asha._get_id(point)] = (objective, point)

    def get_candidate(self, rung_id):
        budget, rung = self.rungs[rung_id]
        next_rung = self.rungs[rung_id + 1][1]

        k = len(rung) // self.reduction_factor
        rung = list(sorted(rung))
        k = min(k, len(rung))

        for i in range(k):
            objective, point = rung[i]
            if point not in next_rung:
                return point, objective

        return None, None

    @property
    def is_done(self):
        return len(self.rungs[0][1])

    def update_rungs(self):
        """

        Notes
        -----
            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.
        """
        # NOTE: There should be base + 1 rungs
        for rung_id in range(len(self.rungs) - 1, 0, -1):
            candidate, objective = self.get_candidate(rung_id)

            if candidate:
                self.rungs[rung_id + 1][1][self.asha._get_id(candidate)] = (objective, candidate)
                candidate[self.asha.fidelity_index] = self.rungs[-1][0]

                return candidate
