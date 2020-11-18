import numpy

from orion.benchmark.base import BaseTask


class CarromTable(BaseTask):

    def __init__(self, max_trials=20):
        self.max_trials = max_trials
        super(CarromTable, self).__init__(max_trials=max_trials)

    def get_blackbox_function(self):
        def carromtable(x):
            """Evaluate a 2-D CarromTable function."""

            a = numpy.exp(2 * numpy.absolute(1 - numpy.sqrt(numpy.square(x[0]) + numpy.square(x[1]))))
            y = - a / 30 * numpy.square(numpy.cos(x[0])) * numpy.square(numpy.cos(x[1]))

            return [dict(
                name='carromtable',
                type='objective',
                value=y)]

        return carromtable

    def get_max_trials(self):
        return self.max_trials

    def get_search_space(self):

        rspace = {'x': 'uniform(-10, 10, shape=2)'}

        return rspace