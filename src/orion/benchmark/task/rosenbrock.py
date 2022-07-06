#!/usr/bin/env python
"""
Task for RosenBrock Function
==============================
"""
import numpy

from orion.benchmark.task.base import BenchmarkTask


class RosenBrock(BenchmarkTask):
    """`RosenBrock function <http://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.Rosenbrock>`_
    as benchmark task"""

    def __init__(self, max_trials=20, dim=2):
        self.dim = dim
        super().__init__(max_trials=max_trials, dim=dim)

    def call(self, x):
        """Evaluate a n-D rosenbrock function."""
        x = numpy.asarray(x)
        summands = 100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
        y = numpy.sum(summands)
        return [dict(name="rosenbrock", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": f"uniform(-5, 10, shape={self.dim})"}

        return rspace
