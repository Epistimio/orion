import pandas as pd
import plotly.graph_objects as go


def regret(experiment, order_by, verbose_hover, **kwargs):

    def template_trials():
        template = '<b>ID: %{customdata[0]}</b><br>' \
            'value: %{y}<br>' \
            'time: %{customdata[1]|%Y-%m-%d %H:%M:%S}<br>'

        if verbose_hover:
            template += 'parameters: %{customdata[2]}'

        template += '<extra></extra>'

        return template

    def template_best():
        return '<b>Best ID: %{customdata[0]}</b><br>' \
            'value: %{customdata[1]}' \
            '<extra></extra>'

    ORDER_KEYS = ['suggested', 'reserved', 'completed']

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    trials = list(filter(lambda trial: trial.status == 'completed', experiment.fetch_trials()))

    data = [(trial.id, trial.status, trial.submit_time, trial.start_time,
             trial.end_time, format_hyperparameters(trial.params), trial.objective.value) for trial in trials]

    df = pd.DataFrame(data, columns=['id', 'status', ORDER_KEYS[0],
                                     ORDER_KEYS[1], ORDER_KEYS[2], 'params', 'objective'])

    df = df.sort_values(order_by)

    df['best'] = df['objective'].cummin()
    df['best_id'] = get_best_ids(df)

    fig = go.Figure()

    fig.add_scatter(y=df['objective'],
                    mode='markers',
                    name='trials',
                    customdata=list(zip(df['id'], df[order_by], df['params'])),
                    hovertemplate=template_trials())
    fig.add_scatter(y=df['best'],
                    mode='lines',
                    name='best-to-date',
                    customdata=list(zip(df['best_id'], df['best'])),
                    hovertemplate=template_best())

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


def format_value(value):
    """
    Hyperparameter can have many types, sometimes they can even be lists.
    If one of the value is a float, it has to be compact.
    """
    if isinstance(value, str):
        return value

    try:
        return f"[{','.join(format_value(x) for x in value)}]"

    except TypeError:
        if isinstance(value, float):
            return f"{value:.5G}"
        else:
            return value


def format_hyperparameters(hyperparameters):
    result = ''

    for name, value in hyperparameters.items():
        x = f'<br>  {name[1:]}: {format_value(value)}'
        result += x

    return result
