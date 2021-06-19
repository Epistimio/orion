"""
Plotly backend for plotting methods
===================================

"""
import functools
from collections import Iterable

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
    experiment,
    with_evc_tree=True,
    model="RandomForestRegressor",
    model_kwargs=None,
    n_points=20,
    n_runs=10,
    **kwargs,
):
    """Plotly implementation of `orion.plotting.lpi`"""
    if not experiment:
        raise ValueError("Parameter 'experiment' is None")

    if model_kwargs is None:
        model_kwargs = {}

    df = experiment.to_pandas(with_evc_tree=with_evc_tree)
    df = df.loc[df["status"] == "completed"]

    if df.empty:
        return go.Figure()

    df = orion.analysis.lpi(
        df,
        experiment.space,
        model=model,
        n_points=n_points,
        n_runs=n_runs,
        **model_kwargs,
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df.index.tolist(),
                y=df["LPI"].tolist(),
                error_y=dict(type="data", array=df["STD"].tolist()),
            )
        ]
    )

    y_axis_label = "Local Parameter Importance (LPI)"
    fig.update_layout(
        title=f"LPI for experiment '{experiment.name}'",
        xaxis_title="Hyperparameters",
        yaxis_title=y_axis_label,
    )

    return fig


def parallel_coordinates(
    experiment, with_evc_tree=True, order=None, colorscale="YlOrRd", **kwargs
):
    """Plotly implementation of `orion.plotting.parallel_coordinates`"""

    def build_frame():
        """Builds the dataframe for the plot"""
        names = list(experiment.space.keys())

        df = experiment.to_pandas(with_evc_tree=with_evc_tree)
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


def rankings(experiments, with_evc_tree=True, order_by="suggested", **kwargs):
    """Plotly implementation of `orion.plotting.rankings`"""

    def reformat_competitions(experiments):
        if isinstance(experiments, dict) and isinstance(
            next(iter(experiments.values())), Iterable
        ):
            competitions = []
            remaining = True
            n_competitions = len(next(iter(experiments.values())))
            for ith_competition in range(n_competitions):
                competition = {}
                for name in experiments.keys():
                    competition[name] = experiments[name][ith_competition]
                competitions.append(competition)
        elif isinstance(experiments, dict):
            competitions = experiments
        elif isinstance(experiments, Iterable) and not isinstance(experiments[0], dict):
            competitions = {
                f"{experiment.name}-v{experiment.version}": experiment
                for experiment in experiments
            }
        else:
            competitions = experiments

        return competitions

    def build_groups(competitions):

        if not isinstance(competitions, dict):
            rankings = []
            for competition in competitions:
                rankings.append(build_frame(competition))
            df = pd.concat(rankings)
            data_frames = orion.analysis.average(
                df, group_by=["order", "name"], key="rank", return_var=True
            )
        else:
            data_frames = build_frame(competitions)

        return data_frames

    def build_frame(competition):
        """Builds the dataframe for the plot"""

        frames = []
        for name, experiment in competition.items():
            df = experiment.to_pandas(with_evc_tree=with_evc_tree)
            df = df.loc[df["status"] == "completed"]
            df = df.sort_values(order_by)
            df = orion.analysis.regret(df)
            df["name"] = [name] * len(df)
            df["order"] = range(len(df))
            frames.append(df)

        df = pd.concat(frames)

        return orion.analysis.ranking(df)

    def get_objective_name(competition):
        """Infer name of objective based on trials of one experiment"""
        if not isinstance(competition, dict):
            return get_objective_name(competition[0])

        for experiment in competition.values():
            trials = experiment.fetch_trials_by_status("completed")
            if trials:
                return trials[0].objective.name
        return "objective"

    ORDER_KEYS = ["suggested", "reserved", "completed"]

    if not experiments:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    competitions = reformat_competitions(experiments)
    df = build_groups(competitions)

    fig = go.Figure()

    if df.empty:
        return fig

    names = set(df["name"])
    for i, name in enumerate(sorted(names)):
        exp_data = df[df["name"] == name]
        if "rank_mean" in exp_data:
            y = exp_data["rank_mean"]
        else:
            y = exp_data["rank"]
        x = list(range(len(y)))
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=px.colors.qualitative.G10[i]),
            name=name,
        )
        if "rank_var" in exp_data:
            dy = exp_data["rank_var"]
            fig.add_scatter(
                x=list(x) + list(x)[::-1],
                y=list(y - dy) + list(y + dy)[::-1],
                fill="toself",
                showlegend=False,
                line=dict(color=px.colors.qualitative.G10[i], width=0),
                name=name,
            )

    objective = get_objective_name(competitions)
    fig.update_layout(
        title=f"Average Rankings",
        xaxis_title=f"Trials ordered by {order_by} time",
        yaxis_title=f"Ranking based on {objective}",
        hovermode="x",
    )
    return fig


def partial_dependencies(
    experiment,
    with_evc_tree=True,
    params=None,
    smoothing=0.85,
    n_grid_points=10,
    n_samples=50,
    colorscale="Blues",
    model="RandomForestRegressor",
    model_kwargs=None,
    verbose_hover=True,
):
    """Plotly implementation of `orion.plotting.partial_dependencies`"""

    def build_data():
        """Builds the dataframe for the plot"""
        df = experiment.to_pandas(with_evc_tree=with_evc_tree)

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
            if "reciprocal" in dim.prior_name or dim.type == "fidelity":
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

    def _plot_scatter(x, y, df):
        return go.Scatter(
            x=x,
            y=y,
            marker={
                "line": {"width": 0.5, "color": "Grey"},
                "color": "black",
                "size": 5,
            },
            mode="markers",
            opacity=0.5,
            showlegend=False,
            customdata=list(zip(df["id"], df["suggested"], df["params"])),
            hovertemplate=_template_trials(verbose_hover),
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
        fig.add_trace(
            _plot_scatter(df[x_name], df["objective"], df),
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
                _plot_scatter(df[x_name], df[y_name], df),
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


def regret(
    experiment, with_evc_tree=True, order_by="suggested", verbose_hover=True, **kwargs
):
    """Plotly implementation of `orion.plotting.regret`"""

    def build_frame():
        """Builds the dataframe for the plot"""
        df = experiment.to_pandas(with_evc_tree=with_evc_tree)

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


def regrets(experiments, with_evc_tree=True, order_by="suggested", **kwargs):
    """Plotly implementation of `orion.plotting.regrets`"""

    compute_average = bool(
        isinstance(experiments, dict)
        and isinstance(next(iter(experiments.values())), Iterable)
    )

    def build_groups():
        """Build dataframes for groups of experiments"""
        # TODO move this
        if compute_average:
            data_frames = dict()
            for name, group in experiments.items():
                df = orion.analysis.average(build_frame(group), return_var=True)
                df["name"] = [name] * len(df)
                data_frames[name] = df
            data_frames = pd.concat(data_frames)
        elif isinstance(experiments, dict):
            data_frames = build_frame(experiments.values(), list(experiments.keys()))
        else:
            data_frames = build_frame(experiments)

        return data_frames

    def build_frame(experiments, names=None):
        """Builds the dataframe for the plot"""
        frames = []
        for i, experiment in enumerate(experiments):
            df = experiment.to_pandas(with_evc_tree=with_evc_tree)
            df = df.loc[df["status"] == "completed"]
            df = df.sort_values(order_by)
            df = orion.analysis.regret(df)
            df["name"] = [
                names[i] if names else f"{experiment.name}-v{experiment.version}"
            ] * len(df)
            df["order"] = range(len(df))
            frames.append(df)

        return pd.concat(frames)

    def get_objective_name(experiments):
        """Infer name of objective based on trials of one experiment"""
        if compute_average and isinstance(experiments, dict):
            return get_objective_name(sum(map(list, experiments.values()), []))

        if isinstance(experiments, dict):
            experiments = experiments.values()

        for experiment in experiments:
            trials = experiment.fetch_trials_by_status("completed")
            if trials:
                return trials[0].objective.name
        return "objective"

    ORDER_KEYS = ["suggested", "reserved", "completed"]

    if not experiments:
        raise ValueError("Parameter 'experiment' is None")

    if order_by not in ORDER_KEYS:
        raise ValueError(f"Parameter 'order_by' is not one of {ORDER_KEYS}")

    df = build_groups()

    fig = go.Figure()

    if df.empty:
        return fig

    names = set(df["name"])
    for i, name in enumerate(sorted(names)):
        exp_data = df[df["name"] == name]
        if "best_mean" in exp_data:
            y = exp_data["best_mean"]
        else:
            y = exp_data["best"]
        x = list(range(len(y)))
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=px.colors.qualitative.G10[i]),
            name=name,
        )
        if "best_var" in exp_data:
            dy = numpy.sqrt(exp_data["best_var"])
            fig.add_scatter(
                x=list(x) + list(x)[::-1],
                y=list(y - dy) + list(y + dy)[::-1],
                fill="toself",
                showlegend=False,
                line=dict(color=px.colors.qualitative.G10[i]),
                name=name,
            )

    fig.update_layout(
        title=f"Average Regret",
        xaxis_title=f"Trials ordered by {order_by} time",
        yaxis_title=get_objective_name(experiments),
        hovermode="x",
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
        x = f"<br>  {name}: {_format_value(value)}"
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
