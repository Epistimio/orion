#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.task` -- Task for Branin Function
================================================================

.. module:: task
   :platform: Unix
   :synopsis: Benchmark algorithms with Branin function.

"""
import math

import numpy

from orion.benchmark.base import BaseTask


class Branin(BaseTask):
    """Branin function as benchmark task"""

    def __init__(self, max_trials=20):
        super(Branin, self).__init__(max_trials=max_trials)

    def get_blackbox_function(self):
        """
        Return the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """
        def branin(x):
            """Evaluate a 2-D branin function."""
            x[0] = (x[1] * 15) - 5
            x[1] = x[0] * 15

            a = 1
            b = 5.1 / (4 * numpy.square(math.pi))
            c = 5 / math.pi
            r = 6
            t = 1 / (8 * math.pi)
            s = 10
            y = a * numpy.square(x[1] - b * numpy.square(x[0]) + c * x[0] - r) + \
                s * (1 - t) * numpy.cos(x[0]) + s

            return [dict(
                name='branin',
                type='objective',
                value=y)]

        return branin

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {'x': 'uniform(0, 1, shape=2)'}

        return rspace
