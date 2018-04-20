# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.primary_algo` -- Sanitizing wrapper of main algorithm
===============================================================================

.. module:: primary_algo
   :platform: Unix
   :synopsis: Performs checks and organizes required transformations of points.

"""

from orion.algo.base import BaseAlgorithm
# TODO Define Transformation classes, request and composite them using
# decorator pattern + `Factory`


class PrimaryAlgo(BaseAlgorithm):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    """

    def __init__(self, space, algorithm_config):
        """
        Initialize the primary algorithm.

        Parameters
        ----------
        space : `orion.algo.space.Space`
           The original definition of a problem's parameters space.
        algorithm_config : dict
           Configuration for the algorithm.

        """
        self.algorithm = None
        super(PrimaryAlgo, self).__init__(space, algorithm=algorithm_config)
        # TODO check requirements
        # TODO cascade Transformation through `Factory`

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = self.algorithm.suggest(num)
        for point in points:
            assert point in self._space  # TODO substitute with transformed space
        # TODO Transform back to the original space
        return points

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        for point in points:
            assert point in self._space
        assert len(points) == len(results)
        # TODO Transform into required space
        self.algorithm.observe(points, results)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.algorithm.is_done

    def score(self, point):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        assert point in self._space
        # TODO Transform into required space
        return self.algorithm.score(point)

    def judge(self, point, measurements):
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        assert point in self._space
        # TODO Transform into required space
        return self.algorithm.judge(point, measurements)

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return self.algorithm.should_suspend

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.algorithm.configuration

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space

    #  @property
    #  def transformed_space(self):
    #      pass
