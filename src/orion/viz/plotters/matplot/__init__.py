# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.matplot` -- Matplot wrapper
====================================================

.. module:: matplot
   :platform: Unix
   :synopsis:

"""

import logging

from matplotlib import pyplot

from orion.viz.plotters.library import Library

log = logging.getLogger(__name__)


class Matplot(Library):
    """Provide a wrapper for plots using the `matplotlib` library. This class will intercept
    and forward calls that correspond to `pyplot` calls.
    """

    implementation_module = "orion.viz.plotters.matplot"

    def __init__(self, analysis, save_format, **plotter_config):
        """Call base class"""
        super(Matplot, self).__init__(analysis, save_format, **plotter_config)

    def __setattr__(self, name, value):
        """Forward the call to `pyplot` if the name corresponds to one of the library's function"""
        if hasattr(pyplot, name):
            proxy_call = getattr(pyplot, name)
            args = value.get('args', [])
            kwargs = value.get('kwargs', {})
            proxy_call(*args, **kwargs)
        else:
            self.__dict__[name] = value
