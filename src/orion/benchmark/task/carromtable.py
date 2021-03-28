#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task for CarromTable Function
==============================
"""
import numpy

from orion.benchmark.task.base import BaseTask


class CarromTable(BaseTask):
    """`CarromTable function <http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CarromTable>`_
    as benchmark task"""

    def __init__(self, max_trials=20):
        super(CarromTable, self).__init__(max_trials=max_trials)

    def call(self, x):
        """Evaluate a 2-D CarromTable function."""
        a = numpy.exp(
            2 * numpy.absolute(1 - numpy.sqrt(numpy.square(x[0]) + numpy.square(x[1])))
        )
        y = -a / 30 * numpy.square(numpy.cos(x[0])) * numpy.square(numpy.cos(x[1]))

        return [dict(name="carromtable", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(-10, 10, shape=2)"}

        return rspace
