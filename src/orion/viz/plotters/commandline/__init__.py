# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.commandline` -- Commandline wrapper
============================================================

.. module:: commandline
   :platform: Unix
   :synopsis:

"""

import logging

from orion.viz.plotters.library import Library

log = logging.getLogger(__name__)


class Commandline(Library):
    """This wrapper serves to maintain compatibility with the structure for library support"""

    implementation_module = "orion.viz.plotters.commandline"

    def __init__(self, analysis, save_format, **plotter_config):
        """Call base class"""
        super(Commandline, self).__init__(analysis, save_format, **plotter_config)
