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

from orion.viz.plotters.base import Library

log = logging.getLogger(__name__)


class Matplot(Library):

    module_addendum = "orion.viz.plotters.matplot"

    def __init__(self, analysis, save_format, **plotter_config):
        super(Matplot, self).__init__(analysis, save_format, instance=plotter_config)

    def __setattr__(self, name, value):
        if hasattr(pyplot, name):
            proxy_call = getattr(pyplot, name)
            proxy_call(**value)
        else:
            self.__dict__[name] = value
