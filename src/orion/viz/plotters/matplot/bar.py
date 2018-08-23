# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.lpi` -- LPI analyser
==============================================

.. module:: lpi
   :platform: Unix
   :synopsis:

"""

import logging

from orion.viz.plotters.base import BasePlotter
from orion.viz.analysis import TimeSeriesAnalysis

from matplotlib import pyplot

log = logging.getLogger(__name__)


class Bar(BasePlotter):

    def __init__(self, analysis, save_formats, **bar_args):
        super(Bar, self).__init__(analysis, save_formats, bar_args=bar_args)

    def plot(self):
        pyplot.bar(range(len(self.analysis.value.keys())), self.analysis.value.values(),
                   **self.bar_args)

        for format_type in self.save_formats:
            pyplot.savefig('title', format=format_type)

        pyplot.clf()

    @property
    def required_analysis(self):
        return [TimeSeriesAnalysis]
