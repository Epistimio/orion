# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.matplot` -- Matplot wrapper
====================================================

.. module:: lpi
   :platform: Unix
   :synopsis:

"""

import logging

from matplotlib import pyplot

from orion.viz.plotters import BasePlotter

log = logging.getLogger(__name__)


class Matplot(BasePlotter):

    module_addendum = "matplot"

    def __init__(self, analysis, save_format, **plotter_config):
        self.plotter = None
        super(Matplot, self).__init__(analysis, save_format, plotter=plotter_config)

    def plot(self):
        return self.plotter.plot()

    @property
    def required_analysis(self):
        return self.plotter.required_analysis

    def __setattr__(self, name, value):
        if hasattr(pyplot, name):
            proxy_call = getattr(pyplot, name)
            proxy_call(**value)
        else:
            self.__dict__[name] = value
