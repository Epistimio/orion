import pandas as pd
import plotly.graph_objects as go


def regret(experiment, order_by='suggested', **kwargs):
    """
    Make a plot to visualize the performance of the hyper-optimization process.

    The x-axis contain the trials and the y-axis their respective best performance.

    Parameters
    ----------
        experiment: ExperimentClient
            The orion object containing the data

        order_by: str
            Indicates how the trials should be ordered. Acceptable options are below.
            See attributes of `Trial` for more details.

            * 'suggested': Sort by trial suggested time (default).
            * 'reserved': Sort by trial reserved time.
            * 'completed': Sort by trial completed time.

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

    ORDER_KEYS = ['suggested', 'reserved', 'completed']

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    trials = list(filter(lambda trial: trial.status == 'completed', experiment.fetch_trials()))

    data = [(trial.id, trial.status, trial.submit_time, trial.start_time,
             trial.end_time, trial.objective.value) for trial in trials]
    df = pd.DataFrame(data, columns=['id', 'status', ORDER_KEYS[0],
                                     ORDER_KEYS[1], ORDER_KEYS[2], 'objective'])

    df = df.sort_values(order_by)

    df['best'] = df['objective'].cummin()
    df['best_id'] = get_best_ids(df)

    fig = go.Figure()

    fig.add_scatter(y=df['objective'],
                    mode='markers',
                    name='trials',
                    customdata=list(zip(df['id'], df[order_by])),
                    hovertemplate="<b>ID: %{customdata[0]}</b><br>value: %{y}<br>time: %{customdata[1]}<extra></extra>")
    fig.add_scatter(y=df['best'],
                    mode='lines',
                    name='best-to-date',
                    customdata=list(zip(df['best_id'], df['best'])),
                    hovertemplate="<b>Best ID: %{customdata[0]}</b><br>value: %{customdata[1]}<extra></extra>")

    y_axis_label = f"{trials[0].objective.type.capitalize()} '{trials[0].objective.name}'"
    fig.update_layout(title=f"Regret for experiment '{experiment.name}'",
                      xaxis_title=f"Trials ordered by {order_by} time",
                      yaxis_title=y_axis_label)

    return fig


def get_best_ids(df):
    best_id = None
    result = []

    for i, id in enumerate(df.id):
        if df.objective[i] == df.best[i]:
            best_id = id
        result.append(best_id)

    return result
