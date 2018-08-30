# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.library` -- Wrapper over `BasePlotter` objects for different libraries
===============================================================================================

.. module:: library
   :platform: Unix
   :synopsis: Wrapper for different `BasePlotter`

"""
from orion.core.utils import Wrapper
from orion.viz.plotters.base import BasePlotter


class Library(Wrapper):
    """Wrapper use create `BasePlotter` objects while providing library-specific calls"""

    implementation_module = "orion.viz.plotters"

    def __init__(self, analysis, save_formats, **plotter_config):
        """Initialize the wrapper and its plotter"""
        super(Library, self).__init__(analysis, save_formats, instance=plotter_config)

    @property
    def wraps(self):
        """Wrap all plotters"""
        return BasePlotter
