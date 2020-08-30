import numpy

from orion.benchmark.base import BaseTask


class RosenBrock(BaseTask):

    def __init__(self, max_trials=20):
        self.max_trials = max_trials
        super(RosenBrock, self).__init__()

    def get_blackbox_function(self):
        def rosenbrock(x):
            """Evaluate a n-D rosenbrock function."""
            x = numpy.asarray(x)
            summands = 100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
            y = numpy.sum(summands)
            return [dict(
                name='rosenbrock',
                type='objective',
                value=y)]

        return rosenbrock

    def get_max_trials(self):
        return self.max_trials

    def get_search_space(self):

        rspace = {'x': 'uniform(-5, 10, shape=2)'}

        return rspace
