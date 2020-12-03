"""
:mod:`orion.core.plotting.backend_plotly` -- Plotly backend for plotting methods
================================================================================

.. module:: backend_plotly
   :platform: Unix
   :synopsis: Plotly backend for plotting methods
"""
import functools

import numpy
import pandas as pd
import plotly.graph_objects as go

import orion.analysis.regret
from orion.algo.space import Categorical, Fidelity


def lpi(experiment, model="RandomForestRegressor", model_kwargs=None, n=20, **kwargs):
    """Plotly implementation of `orion.plotting.lpi`"""
    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if model_kwargs is None:
        model_kwargs = {}

    df = experiment.to_pandas()
    df = orion.analysis.lpi(df, experiment.space, model=model, n=n, **model_kwargs)

    fig = go.Figure(data=[go.Bar(x=df.index.tolist(), y=df["LPI"].tolist())])

    y_axis_label = "Local Parameter Importance (LPI)"
    fig.update_layout(
        title=f"LPI for experiment '{experiment.name}'",
        xaxis_title="Hyperparameters",
        yaxis_title=y_axis_label,
    )

    return fig


def parallel_coordinates(experiment, order=None, colorscale="YlOrRd", **kwargs):
    """Plotly implementation of `orion.plotting.parallel_coordinates`"""

    def build_frame():
        """Builds the dataframe for the plot"""
        names = list(experiment.space.keys())

        df = experiment.to_pandas()
        df = df.loc[df["status"] == "completed"]

        df[names] = df[names].transform(
            functools.partial(_curate_params, space=experiment.space)
        )

        df = _flatten_dims(df, experiment.space)

        return df

    def infer_order(order):
        """Create order if not passed, otherwised verify it"""
        if order is None:
            order = list(experiment.space.keys())
            fidelity_dims = [
                dim for dim in experiment.space.values() if isinstance(dim, Fidelity)
            ]
            if fidelity_dims:
                del order[order.index(fidelity_dims[0].name)]
                order.insert(0, fidelity_dims[0].name)
        else:
            names = set(experiment.space.keys())
            any_invalid = set(order) - names
            if any_invalid:
                raise ValueError(
                    f"Some names are invalid: {any_invalid} not in {names}"
                )

        return order

    def get_dimension(data, name, dim):
        dim_data = dict(label=name, values=data[name])
        if isinstance(dim, Categorical):
            dim_data["tickvals"] = list(range(len(dim.categories)))
            dim_data["ticktext"] = dim.categories
        elif isinstance(dim, Fidelity):
            dim_data["range"] = (dim.low, dim.high)
        else:
            dim_data["range"] = dim.interval()
        return dim_data

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    trial = experiment.fetch_trials_by_status("completed")[0]
    df = build_frame()

    dimensions = []
    for name in infer_order(order):
        dim = experiment.space[name]
        shape = dim.shape
        if shape:
            assert len(shape) == 1
            for i in range(shape[0]):
                name_i = f"{name}[{i}]"
                dimensions.append(get_dimension(df, name_i, dim))
        else:
            dimensions.append(get_dimension(df, name, dim))

    objective_name = trial.objective.name

    objectives = df["objective"]
    omin = min(df["objective"])
    omax = max(df["objective"])

    dimensions.append(dict(label=objective_name, range=(omin, omax), values=objectives))

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=objectives,
                colorscale=colorscale,
                showscale=True,
                cmin=omin,
                cmax=omax,
                colorbar=dict(title=objective_name),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title=f"Parallel Coordinates Plot for experiment '{experiment.name}'"
    )

    return fig


def regret(experiment, order_by, verbose_hover, **kwargs):
    """Plotly implementation of `orion.plotting.regret`"""

    def build_frame():
        """Builds the dataframe for the plot"""
        df = experiment.to_pandas()

        names = list(experiment.space.keys())
        df["params"] = df[names].apply(_format_hyperparameters, args=(names,), axis=1)

        df = df.loc[df["status"] == "completed"]
        df = df.sort_values(order_by)
        df = orion.analysis.regret(df)
        return df

    ORDER_KEYS = ["suggested", "reserved", "completed"]

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    trial = experiment.fetch_trials_by_status("completed")[0]
    df = build_frame()

    fig = go.Figure()

    fig.add_scatter(
        y=df["objective"],
        mode="markers",
        name="trials",
        customdata=list(zip(df["id"], df[order_by], df["params"])),
        hovertemplate=_template_trials(verbose_hover),
    )
    fig.add_scatter(
        y=df["best"],
        mode="lines",
        name="best-to-date",
        customdata=list(zip(df["best_id"], df["best"])),
        hovertemplate=_template_best(),
    )

    y_axis_label = f"{trial.objective.type.capitalize()} '{trial.objective.name}'"
    fig.update_layout(
        title=f"Regret for experiment '{experiment.name}'",
        xaxis_title=f"Trials ordered by {order_by} time",
        yaxis_title=y_axis_label,
    )

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
    result = ""

    for name, value in zip(names, hyperparameters):
        x = f"<br>  {name[1:]}: {_format_value(value)}"
        result += x

    return result


def _template_trials(verbose_hover):
    template = (
        "<b>ID: %{customdata[0]}</b><br>"
        "value: %{y}<br>"
        "time: %{customdata[1]|%Y-%m-%d %H:%M:%S}<br>"
    )

    if verbose_hover:
        template += "parameters: %{customdata[2]}"

    template += "<extra></extra>"

    return template


def _template_best():
    return (
        "<b>Best ID: %{customdata[0]}</b><br>"
        "value: %{customdata[1]}"
        "<extra></extra>"
    )


def _curate_params(data, space):
    dim = space[data.name]
    if isinstance(dim, Categorical):
        data = numpy.array(data.tolist())  # To unpack lists if dim shape > 1
        shape = data.shape
        assert len(shape) <= 2
        idx = numpy.argmax(data.reshape(-1, 1) == numpy.array(dim.categories), axis=1)
        idx = idx.reshape(shape)
        if len(shape) > 1:
            return [list(idx[i]) for i in range(shape[0])]
        return idx
    return data


def _flatten_dims(data, space):
    for key, dim in space.items():
        if dim.shape:
            assert len(dim.shape) == 1

            # expand df.tags into its own dataframe
            values = data[key].apply(pd.Series)

            # rename each hp
            values = values.rename(columns=lambda x: f"{key}[{x}]")

            data = pd.concat([data[:], values[:]], axis=1)

    return data
