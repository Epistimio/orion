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
    def __init__(self, analysis, save_format, plotter_config):
        super(Matplot, self).__init__(analysis, save_format, plotter_config)

    def __setattr__(self, name, value):
        if hasattr(pyplot, name):
            proxy_call = getattr(pyplot, name)
            print(name)
            print(value)
            proxy_call(**value)
