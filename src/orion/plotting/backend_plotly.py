"""
:mod:`orion.core.plotting.backend_plotly` -- Plotly backend for plotting methods
================================================================================

.. module:: backend_plotly
   :platform: Unix
   :synopsis: Plotly backend for plotting methods
"""
import pandas as pd
import plotly.graph_objects as go


def regret(experiment, order_by, verbose_hover, **kwargs):
    """Plotly implementation of `orion.plotting.regret`"""
    def build_frame(trials):
        """Builds the dataframe for the plot"""
        data = [(trial.id, trial.status,
                trial.submit_time, trial.start_time, trial.end_time,
                _format_hyperparameters(trial.params), trial.objective.value) for trial in trials]

        df = pd.DataFrame(data, columns=['id', 'status', ORDER_KEYS[0],
                                         ORDER_KEYS[1], ORDER_KEYS[2], 'params', 'objective'])

        df = df.sort_values(order_by)

        df['best'] = df['objective'].cummin()
        df['best_id'] = _get_best_ids(df)

        return df

    ORDER_KEYS = ['suggested', 'reserved', 'completed']

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    trials = list(filter(lambda trial: trial.status == 'completed', experiment.fetch_trials()))
    df = build_frame(trials)

    fig = go.Figure()

    fig.add_scatter(y=df['objective'],
                    mode='markers',
                    name='trials',
                    customdata=list(zip(df['id'], df[order_by], df['params'])),
                    hovertemplate=_template_trials(verbose_hover))
    fig.add_scatter(y=df['best'],
                    mode='lines',
                    name='best-to-date',
                    customdata=list(zip(df['best_id'], df['best'])),
                    hovertemplate=_template_best())

    y_axis_label = f"{trials[0].objective.type.capitalize()} '{trials[0].objective.name}'"
    fig.update_layout(title=f"Regret for experiment '{experiment.name}'",
                      xaxis_title=f"Trials ordered by {order_by} time",
                      yaxis_title=y_axis_label)

    return fig


def _get_best_ids(dataframe):
    """Links the cumulative best objectives with their respective ids"""
    best_id = None
    result = []

    for i, id in enumerate(dataframe.id):
        if dataframe.objective[i] == dataframe.best[i]:
            best_id = id
        result.append(best_id)

    return result


def _format_value(value):
    """
    Hyperparameter can have many types, sometimes they can even be lists.
    If one of the value is a float, it has to be compact.
    """
    if isinstance(value, str):
        return value

    try:
        return f"[{','.join(_format_value(x) for x in value)}]"

    except TypeError:
        if isinstance(value, float):
            return f"{value:.5G}"
        else:
            return value


def _format_hyperparameters(hyperparameters):
    result = ''

    for name, value in hyperparameters.items():
        x = f'<br>  {name[1:]}: {_format_value(value)}'
        result += x

    return result


def _template_trials(verbose_hover):
    template = '<b>ID: %{customdata[0]}</b><br>' \
        'value: %{y}<br>' \
        'time: %{customdata[1]|%Y-%m-%d %H:%M:%S}<br>'

    if verbose_hover:
        template += 'parameters: %{customdata[2]}'

    template += '<extra></extra>'

    return template


def _template_best():
    return '<b>Best ID: %{customdata[0]}</b><br>' \
        'value: %{customdata[1]}' \
        '<extra></extra>'
