from orion.core.utils import Wrapper
from orion.viz.plotters.library import Library


class PlotterWrapper(Wrapper):
    """Wrapper for different kind of libraries"""

    implementation_module = "orion.viz.plotters"

    def __init__(self, analysis, save_formats, plotter_config):
        super(PlotterWrapper, self).__init__(analysis, save_formats, instance=plotter_config)

        if type(analysis) not in self.required_analysis:
            raise TypeError('Analysis type not supported by this plotter')

    @property
    def wraps(self):
        return Library
