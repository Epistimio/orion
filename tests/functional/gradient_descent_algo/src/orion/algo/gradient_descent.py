# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.gradient_descent` -- Perform gradient descent on a loss surface
================================================================================

.. module:: gradient_descent
   :platform: Unix
   :synopsis: Use gradients to locally search for a minimum.

"""
import numpy

from orion.algo.base import BaseAlgorithm


class Gradient_Descent(BaseAlgorithm):
    """Implement a gradient descent algorithm."""

    def __init__(self, space, learning_rate=1.):
        """Declare `learning_rate` as a hyperparameter of this algorithm."""
        super(Gradient_Descent, self).__init__(space, learning_rate=learning_rate)
        self.has_observed_once = False
        self.current_point = None
        self.gradient = numpy.array([numpy.inf])

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        assert num == 1  # Simple gradient descent does not make sense else
        if not self.has_observed_once:
            return self.space.sample(1)

        self.current_point -= self.learning_rate * self.gradient
        return [self.current_point]

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        self.current_point = numpy.asarray(points[-1])
        self.gradient = numpy.asarray(results[-1]['gradient'])
        self.has_observed_once = True

    @property
    def is_done(self):
        """Implement a terminating condition."""
        return self.learning_rate * numpy.sqrt(self.gradient.dot(self.gradient)) <= 1e-7
