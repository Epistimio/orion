from orion.core.utils import Wrapper
from orion.viz.plotters.base import BasePlotter


class Library(Wrapper):
    """Wrapper use to wrap different plotting libraries"""

    implementation_module = "orion.viz.plotters"

    def __init__(self, analysis, save_formats, **plotter_config):
        """Initialize the wrapper and its plotter"""
        super(Library, self).__init__(analysis, save_formats, instance=plotter_config)

    @property
    def wraps(self):
        """Wrap all plotters"""
        return BasePlotter
