def regret(experiment, **kwargs):
    """
    Make a plot to visualize the performance of the hyper-optimization process.

    The x-axis contain the trials and the y-axis their respective best performance.

    Parameters
    ----------
        experiment: ExperimentClient
            The orion object containing the data

        **kwargs
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

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    import pandas as pd
    import plotly.graph_objects as go

    trials = experiment.fetch_trials()

    data = [(trial.id, trial.status, trial.submit_time, trial.objective.value) for trial in trials]

    df = pd.DataFrame(data, columns=['id', 'status', 'submit', 'objective'])
    df['best'] = df['objective'].cummin()

    fig = go.Figure()

    fig.add_scatter(y=df['objective'],
                    mode='markers',
                    name='objective')

    fig.add_scatter(y=df['best'],
                    mode='lines',
                    name='best')

    fig.update_layout(title="Regret for experiment '{}'".format(experiment.name),
                      xaxis_title="Trials",
                      yaxis_title="Performance")

    return fig


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

    def __call__(self, *args, **kwargs):
        """
        Convenience method to call different types of plots or ExperimentClient.
        """
        raise NotImplementedError()
        # TODO: Delegate call to correct plotting functions.

    def regret(self):
        """ Convenience method for :meth:`orion.plotting.learning`"""
        raise NotImplementedError()
        # TODO: Call self(kind='learning')
