# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.plotters.plotterwrapper` -- Wrapper over `Library` objects for runtime polymorphism
===================================================================================================

.. module:: plotterwrapper
   :platform: Unix
   :synopsis: Wrapper for different ibraries

"""
from orion.core.utils import Wrapper
from orion.viz.plotters.library import Library


class PlotterWrapper(Wrapper):
    """This wrapper keeps an instance of a specific type of library wrapper so that
    they can be instantiated at runtime through the configuration file. Here is a little
    graph to explain the difference between the two of them.

    ##################       #############       ###############
    #                #       #           #       #             #
    # PlotterWrapper # wraps #  Library  # wraps # BasePlotter #
    #    (Wrapper)   #------># (Wrapper) #------>#  (Concept)  #
    #                #       #           #       #             #
    ##################       #############       ###############

    You can see that `PlotterWrapper` creates an instance of a library specific
    wrapper which will then create an instance of a `BasePlotter` object using library
    specific calls. This methods ensure that each library can be separated into their own
    folders will maintaining the ability to be discovered and instantiated through the factory.
    """

    implementation_module = "orion.viz.plotters"

    def __init__(self, analysis, save_formats, plotter_config):
        """Call the base class and make sure  the analysis passed as argument is valid"""
        super(PlotterWrapper, self).__init__(analysis, save_formats, instance=plotter_config)

        assert type(analysis) in self.required_analysis

    @property
    def wraps(self):
        """Wrap :mod:`orion.viz.plotters.library.Library` objects"""
        return Library
