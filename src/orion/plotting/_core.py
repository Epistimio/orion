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

    trials = list(filter(lambda trial: trial.status == 'completed', experiment.fetch_trials()))

    data = [(trial.id, trial.status, trial.submit_time, trial.objective.value) for trial in trials]
    df = pd.DataFrame(data, columns=['id', 'status', 'submit', 'objective'])
    df['best'] = df['objective'].cummin()
    df['best_id'] = get_best_ids(df)

    fig = go.Figure()

    fig.add_scatter(y=df['objective'],
                    mode='markers',
                    name='trials',
                    customdata=list(zip(df['id'], df['submit'])),
                    hovertemplate="<b>ID: %{customdata[0]}</b><br>value: %{y}<br>submitted: %{customdata[1]}<extra></extra>")
    fig.add_scatter(y=df['best'],
                    mode='lines',
                    name='best-to-date',
                    customdata=list(zip(df['best_id'], df['best'])),
                    hovertemplate="<b>Best ID: %{customdata[0]}</b><br>value: %{customdata[1]}<extra></extra>")

    y_axis_label = f"{trials[0].objective.type.capitalize()} '{trials[0].objective.name}'"
    fig.update_layout(title=f"Regret for experiment '{experiment.name}'",
                      xaxis_title="Trials by submit order",
                      yaxis_title=y_axis_label)

    return fig


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


def get_best_ids(df):
    best_id = None
    result = []

    for i, id in enumerate(df.id):
        if df.objective[i] == df.best[i]:
            best_id = id
        result.append(best_id)

    return result
