from orion.benchmark.assessment.base import BaseAssess
from orion.plotting.base import regrets
from typing import List, Tuple, Dict
from orion.core.worker.experiment import Experiment
from orion.client import ExperimentClient
from plotly.graph_objects import Figure
from .averageresult import AverageResult, regrets
from .averagerank import AverageRank, rankings
from collections import defaultdict
import plotly.graph_objs as go
from orion.core.worker.trial import Trial
import plotly
from pandas.core.groupby import DataFrameGroupBy
from typing import Callable
import pandas as pd
import orion.analysis

WarmStartExperimentsTuple = Tuple[ExperimentClient, ExperimentClient, ExperimentClient]


class WarmStartEfficiency(BaseAssess):
    """
    TODO: Evaluate the 'warm start efficiency' (objective value) for each search algorithm
    at different time steps (trial number).
    """

    def __init__(self, task_num: int = 1):
        super().__init__(task_num=task_num)

    def analysis(
        self, task: str, experiments: Dict[int, List[WarmStartExperimentsTuple]]
    ) -> Figure:
        """
        Generate a `plotly.graph_objects.Figure` to display the performance analysis
        based on the assessment purpose.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to
            run for this assessment, and experiment is an instance of
            `orion.core.worker.experiment.Experiment`.
        """

        # TODO: Reuse the figures form other Assessments, but group by cold/warm/hot in
        # addition to by algorithm
        all_plots = []
        experiments_dict: Dict[str, Dict[str, List[ExperimentClient]]] = {}
        for algo_index, list_of_exp_tuples in experiments.items():

            algo_experiments = defaultdict(list)

            for repetition_index, experiment_tuple in enumerate(list_of_exp_tuples):
                for stage, exp in zip(["cold", "warm", "hot"], experiment_tuple):
                    algorithm_name = list(exp.configuration["algorithms"].keys())[0]
                    algo_experiments[stage].append(exp)
                    # figure_experiments_dict[stage].append(exp)

            experiments_dict[algorithm_name] = algo_experiments

            algo_regrets_plot = regrets(algo_experiments)
            all_plots.append(algo_regrets_plot)

            algo_rankings_plot = rankings(algo_experiments)
            all_plots.append(algo_rankings_plot)

        for algorithm_name, algo_experiments in experiments_dict.items():
            cold_start_experiments = algo_experiments["cold"]
            warm_start_experiments = algo_experiments["warm"]
            hot_start_experiments = algo_experiments["hot"]

            warm_start_comparison_figs = warm_start_comparison_figure(
                cold_start_experiments=cold_start_experiments,
                warm_start_experiments=warm_start_experiments,
                hot_start_experiments=hot_start_experiments,
                algorithm_name=algorithm_name,
                task_name=task,
            )
            all_plots.extend(warm_start_comparison_figs)
            # warm_start_comparison_figs.show()

        return all_plots


def build_frame(
    experiments: List[ExperimentClient],
    names: List[str] = None,
    with_evc_tree: bool = True,
) -> pd.DataFrame:
    """Builds the dataframe for the plot"""
    # NOTE: Using a dict instead of a list, and using the experiment ID as the level
    # for the multi-index.
    frames: Dict[str, pd.DataFrame] = {}
    for i, experiment in enumerate(experiments):
        df = experiment.to_pandas(with_evc_tree=with_evc_tree)
        df = df.loc[df["status"] == "completed"]

        # df = df.sort_values(order_by)
        df = orion.analysis.regret(df)

        df["name"] = [
            names[i] if names else f"{experiment.name}-v{experiment.version}"
        ] * len(df)
        df["order"] = range(len(df))
        assert experiment.id not in frames
        # NOTE: This isn't the same as the trial id. It might be the same as `order`
        # though. Not 100% sure.
        df.index.name = "trial_index"
        frames[experiment.id] = df

    stacked_df = pd.concat(frames, names=["experiment_id"])
    assert stacked_df.index.names == ["experiment_id", "trial_index"]
    return stacked_df


def warm_start_comparison_figure(
    cold_start_experiments: List[ExperimentClient],
    warm_start_experiments: List[ExperimentClient],
    hot_start_experiments: List[ExperimentClient],
    algorithm_name: str,
    task_name: str,
    with_evc_tree: bool = True,
) -> List[plotly.graph_objects.Figure]:
    # import dash
    # import dash_core_components as dcc
    # import dash_html_components as html
    # from dash.dependencies import Input, Output
    # import plotly.express as px
    # import plotly.express as px

    # df = px.data.tips()
    # df = TODO
    import pandas as pd
    import orion.analysis

    # These dataframes have a multiindex
    cold_start_df = build_frame(cold_start_experiments, with_evc_tree=with_evc_tree)
    warm_start_df = build_frame(warm_start_experiments, with_evc_tree=with_evc_tree)
    hot_start_df = build_frame(hot_start_experiments, with_evc_tree=with_evc_tree)

    def _check_experiment_ids_dont_overlap(
        cold_start_df: pd.DataFrame,
        warm_start_df: pd.DataFrame,
        hot_start_df: pd.DataFrame,
    ) -> None:
        # Double-check that there is no overlap in the experiment ids between the
        # cold/warm/hot groups.
        cold_experiment_ids = set(v[0] for v in cold_start_df.index)
        warm_experiment_ids = set(v[0] for v in warm_start_df.index)
        hot_experiment_ids = set(v[0] for v in hot_start_df.index)

        unique_experiment_ids = cold_experiment_ids.union(
            warm_experiment_ids, hot_experiment_ids
        )
        number_of_distinct_ids = len(unique_experiment_ids)
        number_of_experiments = (
            len(cold_experiment_ids)
            + len(warm_experiment_ids)
            + len(hot_experiment_ids)
        )
        assert (
            number_of_distinct_ids == number_of_experiments
        ), "some experiment ids are shared across the cold/warm/hot groups!"

    _check_experiment_ids_dont_overlap(cold_start_df, warm_start_df, hot_start_df)

    # Add a new column to indicate the type of warm-start for each experiment.
    cold_start_df["warm_start_type"] = ["cold"] * len(cold_start_df)
    warm_start_df["warm_start_type"] = ["warm"] * len(warm_start_df)
    hot_start_df["warm_start_type"] = ["hot"] * len(hot_start_df)

    df = pd.concat(
        {"cold": cold_start_df, "warm": warm_start_df, "hot": hot_start_df,},
        names=["warm_start_type"],
    )
    assert df.index.names == ["warm_start_type", "experiment_id", "trial_index"]

    # df.drop(columns=["order", "experiment_id"], inplace=True)

    # 'hack' to keep the multi-index, but group by the experiment id.
    # NOTE: There is probably a built-in way to do this.
    # mean_per_experiment = df[df.groupby(level="experiment_id").mean().astype(bool)]

    # NOTE: This *almost* works, but it removes the 'warm-start-type' column, since it's
    # not numeric. (Using `mean(numeric_only=False)`` doesnt work either).
    # mean_per_experiment = df.groupby(level=["warm_start_type", "experiment_id"]).mean()

    # al
    average_over_time = df.groupby(
        level=["warm_start_type", "trial_index"], sort=True
    ).mean()
    color_dict = {"cold": "blue", "warm": "orange", "hot": "red"}

    import plotly.express as px
    import plotly.graph_objects as go
    import plotly

    # TODO: For now, this displays the average objective across all trials across all
    # experiments, for each type of 'warm-starting'.
    figures = []
    y_columns = ["objective", "best"]

    for y_column in y_columns:
        fig = px.box(
            df,
            x="warm_start_type",
            y=y_column,
            color="warm_start_type",
            color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
            title=(
                f"Warm start efficiency - Algo: {algorithm_name}, Task: {task_name} -"
                f"'{y_column}' of all trials from all experiments."
            ),
            points="all",
        )
        figures.append(fig)

    # Group by experiment, taking the minimum across all trials
    grouped_df: pd.DataFrame = df.groupby(
        level=["warm_start_type", "experiment_id"]
    )
    reduction = "mean"

    reduction_dict: Dict[str, Callable[[DataFrameGroupBy], pd.DataFrame]] = {
        "mean": lambda gdf: gdf.mean(),
        "min": lambda gdf: gdf.min(),
    }
    for reduction, reduction_fn in reduction_dict.items():
        plot_df = reduction_fn(grouped_df)
        plot_df.rename(
            columns={k: f"{k}_{reduction}" for k in y_columns},
            inplace=True,
        )
        for y_column in y_columns:
            fig = px.box(
                plot_df,
                x=[v[0] for v in plot_df.index],
                y=f"{y_column}_{reduction}",
                # Use the 'warm_start_type" to determine the colour:
                color=[v[0] for v in plot_df.index],
                color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
                title=(
                    f"Warm start efficiency - Algo: {algorithm_name}, Task: {task_name} -"
                    f"{reduction.capitalize()} of '{y_column}' for each experiment."
                ),
                points="all",
            )
            figures.append(fig)

    return figures
