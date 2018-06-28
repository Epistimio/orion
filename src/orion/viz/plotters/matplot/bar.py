# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.lpi` -- LPI analyser
==============================================

.. module:: lpi
   :platform: Unix
   :synopsis:

"""

import logging

from orion.viz.plotters.matplot import Matplot
from orion.viz.analysis import TimeSeriesAnalysis

from matplotlib import pyplot

log = logging.getLogger(__name__)


class Bar(Matplot):

    def __init__(self, analysis, save_format, **bar_args):
        super(Bar, self).__init__(analysis, save_format=save_format, bar_args=bar_args)

    def plot(self):
        pyplot.bar(range(len(self.analysis.keys())), self.value.values(), **self.bar_args)
        pyplot.clf()

    @property
    def required_analysis(self):
        return [TimeSeriesAnalysis]
