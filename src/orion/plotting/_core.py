import orion.plotting.backend_plotly as backend


def regret(experiment, order_by='suggested', **kwargs):
    return backend.regret(experiment, order_by, **kwargs)


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
        """ Makes a regret plot."""
        return self(kind="regret", **kwargs)
