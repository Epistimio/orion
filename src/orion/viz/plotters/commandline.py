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
from orion.viz.analysis import Analysis

log = logging.getLogger(__name__)


class Commandline(BasePlotter):
    """Prints analysis"""

    def __init__(self, analysis, save_formats):
        super(Commandline, self).__init__(analysis, save_formats)

    def plot(self):
        print(self.analysis.value)

    @property
    def required_analysis(self):
        return [subclass for subclass in Analysis.__subclasses__()]
