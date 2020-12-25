#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.task` -- Task for EggHolder Function
================================================================

.. module:: task
   :platform: Unix
   :synopsis: Benchmark algorithms with EggHolder function.

"""
import numpy

from orion.benchmark.base import BaseTask


class EggHolder(BaseTask):
    """EggHolder function as benchmark task"""

    def __init__(self, max_trials=20, dim=2):
        self.dim = dim
        super(EggHolder, self).__init__(max_trials=max_trials, dim=dim)

    def get_blackbox_function(self):
        """
        Return the black box function to optimize, the function will expect hyper-parameters to
        search and return objective values of trial with the hyper-parameters.
        """
        def eggholder(x):
            """Evaluate a n-D eggholder function."""
            x = numpy.asarray(x)

            a = numpy.square(numpy.absolute(x[:-1] - x[1:] - 47))
            b = numpy.square(numpy.absolute(0.5 * x[:-1] + x[1:] + 47))
            summands = -x[:-1] * numpy.sin(a) - (x[1:] + 47) * numpy.sin(b)

            y = numpy.sum(summands)

            return [dict(
                name='eggholder',
                type='objective',
                value=y)]

        return eggholder

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {'x': 'uniform(-512, 512, shape={})'.format(self.dim)}

        return rspace
