#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task for EggHolder Function
=============================
"""
import numpy

from orion.benchmark.task.base import BaseTask


class EggHolder(BaseTask):
    """`EggHolder function <http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.EggHolder>`_
    as benchmark task"""

    def __init__(self, max_trials=20, dim=2):
        self.dim = dim
        super(EggHolder, self).__init__(max_trials=max_trials, dim=dim)

    def call(self, x):
        """Evaluate a n-D eggholder function."""
        x = numpy.asarray(x)

        a = numpy.square(numpy.absolute(x[:-1] - x[1:] - 47))
        b = numpy.square(numpy.absolute(0.5 * x[:-1] + x[1:] + 47))
        summands = -x[:-1] * numpy.sin(a) - (x[1:] + 47) * numpy.sin(b)

        y = numpy.sum(summands)

        return [dict(name="eggholder", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(-512, 512, shape={})".format(self.dim)}

        return rspace
