# -*- coding: utf-8 -*-
"""
:mod:`metaopt.algo.random` -- Random sampler as optimization algorithm
======================================================================

.. module:: random
   :platform: Unix
   :synopsis: Draw and deliver samples from prior defined in problem's domain.

"""

from metaopt.algo.base import BaseAlgorithm


class Random(BaseAlgorithm):
    """Implement a algorithm that samples randomly from the problem's space."""

    def __init__(self, space):
        """Random sampler takes no other hyperparameter that the problem's space
        itself.
        """
        super(Random, self).__init__(space)

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `metaopt.algo.space.Space`.
        """
        return self.space.sample(num)

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        pass
