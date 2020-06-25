"""
:mod:`orion.plotting.base` -- Provides public plotting API
===========================================================

.. module:: base
   :platform: Unix
   :synopsis: Provides public plotting API

"""
import orion.plotting.backend_plotly as backend


def regret(experiment, order_by='suggested', verbose_hover=False, **kwargs):
    """
    Make a plot to visualize the performance of the hyper-optimization process.

    The x-axis contain the trials and the y-axis their respective best performance.

    Parameters
    ----------
    experiment: ExperimentClient, Experiment or ExperimentView
        The orion object containing the experiment data

    order_by: str
        Indicates how the trials should be ordered. Acceptable options are below.
        See attributes of ``Trial`` for more details.

        * 'suggested': Sort by trial suggested time (default).
        * 'reserved': Sort by trial reserved time.
        * 'completed': Sort by trial completed time.

    verbose_hover: bool
        Indicates whether to display the hyperparameter in hover tooltips. False by default.

    kwargs: dict
        All other plotting keyword arguments to be passed to
        :meth:`plotly.express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided.

    """
    return backend.regret(experiment, order_by, verbose_hover, **kwargs)


PLOT_METHODS = {'regret': regret}


class PlotAccessor:
    """
    Make plots of ExperimentClient.

    Parameters
    ----------
    experiment : ExperimentClient
        The object for which the method is called.

    Raises
    ------
    ValueError
        If no experiment is provided.
    """

    def __init__(self, experiment):
        if not experiment:
            raise ValueError("Parameter 'experiment' is None")
        self._experiment = experiment

    def __call__(self, **kwargs):
        """
        Make different kinds of plots of ExperimentClient.

        Parameters
        ----------
        kind : str
            The kind of plot to produce:

            - 'regret' : Regret plot (default)
        """
        kind = kwargs.pop('kind', 'regret')

        return PLOT_METHODS[kind](self._experiment, **kwargs)

    def regret(self, **kwargs):
        """Make a plot to visualize the performance of the hyper-optimization process."""
        __doc__ = regret.__doc__
        return self(kind="regret", **kwargs)
