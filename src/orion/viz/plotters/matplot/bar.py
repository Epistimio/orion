# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.matplot.bar` -- A bar plotter using `matplotlib`
========================================================================

.. module:: bar
   :platform: Unix
   :synopsis:

"""

import logging

from matplotlib import pyplot

from orion.viz.analysis import CategoricalAnalysis
from orion.viz.plotters.base import BasePlotter

log = logging.getLogger(__name__)


class Bar(BasePlotter):
    """Create a bar plot using the provided analysis through `matplotlib`"""

    def __init__(self, analysis, save_formats, **bar_args):
        """Call base class"""
        super(Bar, self).__init__(analysis, save_formats, bar_args=bar_args)

    def plot(self):
        """Plot the analysis and save it to every format inside `save_formats`"""
        pyplot.bar(range(len(self.analysis.value.keys())), self.analysis.value.values(),
                   **self.bar_args)

        for format_type in self.save_formats:
            pyplot.savefig('title', format=format_type)

        pyplot.clf()

    @property
    def required_analysis(self):
        """Plot `CategoricalAnalsysis` only"""
        return [CategoricalAnalysis]
