# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.commandline.debug` -- Default plotter for debug purposes
=================================================================================

.. module:: commandline
   :platform: Unix
   :synopsis: Prints the analysis to the commandline

"""

import logging

from orion.viz.analysis import Analysis
from orion.viz.plotters.base import BasePlotter

log = logging.getLogger(__name__)


class Debug(BasePlotter):
    """This class can receive any analysis and will simply print its data to the commandline"""

    def __init__(self, analysis, save_formats, **kwargs):
        """Call base class"""
        super(Debug, self).__init__(analysis, save_formats, **kwargs)

    def plot(self):
        """Print the analysis"""
        print(self.analysis.value)

    @property
    def required_analysis(self):
        """Handle every type inheritating from :mod:`orion.viz.analysis.Analysis`"""
        return [subclass for subclass in Analysis.__subclasses__()]
