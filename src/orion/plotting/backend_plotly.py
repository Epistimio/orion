"""
:mod:`orion.core.plotting.backend_plotly` -- Plotly backend for plotting methods
================================================================================

.. module:: backend_plotly
   :platform: Unix
   :synopsis: Plotly backend for plotting methods
"""
import pandas as pd
import plotly.graph_objects as go

import orion.analysis.regret


def regret(experiment, order_by, verbose_hover, **kwargs):
    """Plotly implementation of `orion.plotting.regret`"""
    def build_frame():
        """Builds the dataframe for the plot"""
        df = experiment.to_pandas()

        names = list(experiment.space.keys())
        df['params'] = df[names].apply(
            _format_hyperparameters,
            args=(names, ), axis=1)

        df = df.loc[df['status'] == 'completed']
        df = df.sort_values(order_by)
        df = orion.analysis.regret(df)
        return df

    ORDER_KEYS = ['suggested', 'reserved', 'completed']

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    trial = experiment.fetch_trials_by_status('completed')[0]
    df = build_frame()

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

    y_axis_label = f"{trial.objective.type.capitalize()} '{trial.objective.name}'"
    fig.update_layout(title=f"Regret for experiment '{experiment.name}'",
                      xaxis_title=f"Trials ordered by {order_by} time",
                      yaxis_title=y_axis_label)

    return fig


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


def _format_hyperparameters(hyperparameters, names):
    result = ''

    for name, value in zip(names, hyperparameters):
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
