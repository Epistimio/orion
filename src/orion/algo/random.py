# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.random` -- Random sampler as optimization algorithm
======================================================================

.. module:: random
   :platform: Unix
   :synopsis: Draw and deliver samples from prior defined in problem's domain.

"""
import numpy

from orion.algo.base import BaseAlgorithm


class Random(BaseAlgorithm):
    """Implement a algorithm that samples randomly from the problem's space."""

    def __init__(self, space, seed=None):
        """Random sampler takes no other hyperparameter than the problem's space
        itself.

        :param space: `orion.algo.space.Space` of optimization.
        :param seed: Integer seed for the random number generator.
        """
        super(Random, self).__init__(space, seed=seed)

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
        return self.space.sample(num, seed=tuple(self.rng.randint(0, 1000000, size=3)))

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        pass
