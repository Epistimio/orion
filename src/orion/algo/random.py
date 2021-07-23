# -*- coding: utf-8 -*-
"""
Random sampler as optimization algorithm
========================================

Draw and deliver samples from prior defined in problem's domain.

"""
import numpy

from orion.algo.base import BaseAlgorithm


class Random(BaseAlgorithm):
    """An algorithm that samples randomly from the problem's space.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    """

    def __init__(self, space, seed=None):
        super(Random, self).__init__(space, seed=seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict = super(Random, self).state_dict
        _state_dict["rng_state"] = self.rng.get_state()
        return _state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(Random, self).set_state(state_dict)
        self.seed_rng(0)
        self.rng.set_state(state_dict["rng_state"])

    def suggest(self, num):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = []
        while len(points) < num and not self.is_done:
            seed = tuple(self.rng.randint(0, 1000000, size=3))
            new_point = self.space.sample(1, seed=seed)[0]
            if not self.has_suggested(new_point):
                self.register(new_point)
                points.append(new_point)

        return points
