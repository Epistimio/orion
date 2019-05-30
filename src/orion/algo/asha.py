# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.asha` -- TODO
======================================================================

.. module:: asha 
   :platform: Unix
   :synopsis: TODO

"""
import numpy

from orion.algo.base import BaseAlgorithm


class ASHA(BaseAlgorithm):
    """Implement a algorithm that samples randomly from the problem's space."""

    def __init__(self, space, seed=None, max_resources=100, grace_period=0, reduction_factor=4,
                 brackets=1):
        """Random sampler takes no other hyperparameter than the problem's space
        itself.

        :param space: `orion.algo.space.Space` of optimization.
        :param seed: Integer seed for the random number generator.
        """
        super(ASHA, self).__init__(
            space, seed=seed, max_resources=max_resources, grace_period=grace_period,
            reduction_factor=reduction_factor, brackets=brackets)

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

        # TODO: Add fidelity here based on `space`
        point = self.space.sample(num, seed=self.rng.randint(0, 10000))

        sizes = numpy.array([len(b.rungs) for b in self.brackets])
        probs = numpy.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = numpy.random.choice(len(self.brackets), p=normalized)
        #  TODO: Hash point (without fidelity) to get a unique trial id
        self.trial_info[point] = self.brackets[idx]
        # NOTE: The point is not registered in bracket here, until in `observe()`

        return point

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """


        for point, result in zip(points, results):

            # TODO find which bracket containts the point. If none,
            #      randomly select one. WARNING, the fidelity may not match the bracket, test for it
            bracket.register(point, result)

    def is_done(self):
        return all(bracket.is_done for bracket in self.brackets)


class _Bracket():
    def __init__(self, asha, min_t, max_t, reduction_factor, s):
        self.asha = asha
        self.reduction_factor = reduction_factor
        max_rungs = int(numpy.log(max_t / min_t) / numpy.log(reduction_factor) - s + 1)
        self.rungs = [(min(min_t * reduction_factor**(k + s), max_t), set())
                      for k in reversed(range(max_rungs + 1))]

        if self.rungs[0][0] == self.rungs[1][0]:
            del self.rungs[0]

    def register(self, point, objective):
        self.rungs[-1][1].add((objective, point))

    def get_candidate(self, rung_id):
        budget, rung = self.rungs[rung_id]
        next_rung = self.rungs[rung_id - 1][1]

        k = len(rung) // self.reduction_factor
        rung = list(sorted(rung))
        i = 0
        k = min(k, len(rung))
        while i < k:
            objective, point = rung[i]
            if point not in next_rung:
                return point, objective
            i += 1

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
        for rung_id in range(1, len(self.rungs)):
            candidate, objective = self.get_candidate(rung_id)

            if candidate:
                self.rungs[rung_id - 1][1].add(candidate)

                return candidate
