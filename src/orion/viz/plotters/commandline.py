# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.commandline` -- Default plotter for debug purposes
===========================================================================

.. module:: commandline
   :platform: Unix
   :synopsis: Prints the analysis to the commandline

"""

import logging

from orion.viz.plotters import BasePlotter
from orion.viz.analysis import SingleValueAnalysis

log = logging.getLogger(__name__)


class Commandline(BasePlotter):
    """Prints analysis"""

    def __init__(self, analysis):
        super(Commandline, self).__init__(analysis)

    def plot(self):
        print(self.analysis.value)

    @property
    def required_analysis(self):
        return [SingleValueAnalysis]
