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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import orion.analysis
import orion.analysis.base
from orion.algo.space import Categorical, Fidelity
from orion.core.worker.transformer import build_required_space


def lpi(
    experiment, model="RandomForestRegressor", model_kwargs=None, n_points=20, **kwargs
):
    """Plotly implementation of `orion.plotting.lpi`"""
    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if model_kwargs is None:
        model_kwargs = {}

    df = experiment.to_pandas()
    df = df.loc[df["status"] == "completed"]
    df = orion.analysis.lpi(
        df, experiment.space, model=model, n_points=n_points, **model_kwargs
    )

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

        if df.empty:
            return df

        df[names] = df[names].transform(
            functools.partial(_curate_params, space=experiment.space)
        )

        df = _flatten_dims(df, experiment.space)

        return df

    def infer_order(space, order):
        """Create order if not passed, otherwise verify it"""
        params = orion.analysis.base.flatten_params(space, order)
        if order is None:
            fidelity_dims = [
                dim for dim in experiment.space.values() if isinstance(dim, Fidelity)
            ]
            fidelity = fidelity_dims[0].name if fidelity_dims else None
            if fidelity in params:
                del params[params.index(fidelity)]
                params.insert(0, fidelity)

        return params

    def get_dimension(data, name, dim):
        dim_data = dict(label=name, values=data[name])
        if dim.type == "categorical":
            categories = dim.interval()
            dim_data["tickvals"] = list(range(len(categories)))
            dim_data["ticktext"] = categories
        else:
            dim_data["range"] = dim.interval()
        return dim_data

    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    df = build_frame()

    if df.empty:
        return go.Figure()

    trial = experiment.fetch_trials_by_status("completed")[0]

    flattened_space = build_required_space(
        experiment.space, shape_requirement="flattened"
    )

    dimensions = [
        get_dimension(df, name, flattened_space[name])
        for name in infer_order(experiment.space, order)
    ]

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


def partial_dependencies(
    experiment,
    params=None,
    smoothing=0.85,
    n_grid_points=10,
    n_samples=50,
    colorscale="Blues",
    model="RandomForestRegressor",
    model_kwargs=None,
):
    """Plotly implementation of `orion.plotting.partial_dependencies`"""

    def build_data():
        """Builds the dataframe for the plot"""
        df = experiment.to_pandas()

        names = list(experiment.space.keys())
        df["params"] = df[names].apply(_format_hyperparameters, args=(names,), axis=1)

        df = df.loc[df["status"] == "completed"]
        data = orion.analysis.partial_dependency(
            df,
            experiment.space,
            params=params,
            model=model,
            n_grid_points=n_grid_points,
            n_samples=n_samples,
            **model_kwargs,
        )
        df = _flatten_dims(df, experiment.space)
        return (df, data)

    def _set_scale(figure, dims, x, y):
        for axis, dim in zip("xy", dims):
            if "reciprocal" in dim.prior_name:
                getattr(figure, f"update_{axis}axes")(type="log", row=y, col=x)

    def _plot_marginalized_avg(data, x_name):
        return go.Scatter(
            x=data[0][x_name],
            y=data[1],
            mode="lines",
            name=None,
            showlegend=False,
            line=dict(
                color=px.colors.qualitative.D3[0],
            ),
        )

    def _plot_marginalized_std(data, x_name):
        return go.Scatter(
            x=list(data[0][x_name]) + list(data[0][x_name])[::-1],
            y=list(data[1] - data[2]) + list((data[1] + data[2]))[::-1],
            mode="lines",
            name=None,
            fill="toself",
            showlegend=False,
            line=dict(
                color=px.colors.qualitative.D3[0],
                width=0,
            ),
        )

    def _plot_contour(data, x_name, y_name):
        return go.Contour(
            x=data[0][x_name],
            y=data[0][y_name],
            z=data[1],
            connectgaps=True,
            # Share the same color range across contour plots
            coloraxis="coloraxis",
            line_smoothing=smoothing,
            # To show labels
            contours=dict(
                coloring="heatmap",
                showlabels=True,  # show labels on contours
                labelfont=dict(  # label font properties
                    size=12,
                    color="white",
                ),
            ),
        )

    def _plot_scatter(x, y):
        return go.Scatter(
            x=x,
            y=y,
            marker={"line": {"width": 0.5, "color": "Grey"}, "color": "black"},
            mode="markers",
            showlegend=False,
        )

    if model_kwargs is None:
        model_kwargs = {}

    df, data = build_data()

    if not data:
        return go.Figure()

    params = [
        param_names for param_names in data.keys() if isinstance(param_names, str)
    ]

    flattened_space = build_required_space(
        experiment.space,
        shape_requirement="flattened",
    )

    fig = make_subplots(
        rows=len(params),
        cols=len(params),
        shared_xaxes=True,
        shared_yaxes=False,
    )

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    cmin = float("inf")
    cmax = -float("inf")

    for x_i in range(len(params)):
        x_name = params[x_i]
        fig.add_trace(
            _plot_marginalized_avg(data[x_name], x_name),
            row=x_i + 1,
            col=x_i + 1,
        )
        fig.add_trace(
            _plot_marginalized_std(data[x_name], x_name),
            row=x_i + 1,
            col=x_i + 1,
        )

        _set_scale(fig, [flattened_space[x_name]], x_i + 1, x_i + 1)

        fig.update_xaxes(title_text=x_name, row=len(params), col=x_i + 1)
        if x_i > 0:
            fig.update_yaxes(title_text=x_name, row=x_i + 1, col=1)
        else:
            fig.update_yaxes(title_text="Objective", row=x_i + 1, col=x_i + 1)

        for y_i in range(x_i + 1, len(params)):
            y_name = params[y_i]
            fig.add_trace(
                _plot_contour(
                    data[(x_name, y_name)],
                    x_name,
                    y_name,
                ),
                row=y_i + 1,
                col=x_i + 1,
            )
            fig.add_trace(
                _plot_scatter(df[x_name], df[y_name]),
                row=y_i + 1,
                col=x_i + 1,
            )

            cmin = min(cmin, data[(x_name, y_name)][1].min())
            cmax = max(cmax, data[(x_name, y_name)][1].max())

            _set_scale(
                fig,
                [flattened_space[name] for name in [x_name, y_name]],
                x_i + 1,
                y_i + 1,
            )

    for x_i in range(len(params)):
        plot_id = len(params) * x_i + x_i + 1
        if plot_id > 1:
            key = f"yaxis{plot_id}_range"
        else:
            key = "yaxis_range"
        fig.update_layout(**{key: [cmin, cmax]})

    fig.update_layout(
        title=f"Partial dependencies for experiment '{experiment.name}'",
    )
    fig.layout.coloraxis.colorbar.title = "Objective"

    fig.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=False)

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

    df = build_frame()

    fig = go.Figure()

    if df.empty:
        return fig

    trial = experiment.fetch_trials_by_status("completed")[0]

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

    if trial is None:
        y_axis_label = "Objective unknown"
    else:
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
