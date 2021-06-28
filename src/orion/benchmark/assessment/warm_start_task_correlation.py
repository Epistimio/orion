from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union, Sequence

import orion.analysis
import pandas as pd
import plotly
import plotly.graph_objs as go
from orion.benchmark.assessment.base import BaseAssess
from orion.benchmark.task import BaseTask
from orion.client import ExperimentClient
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.plotting.base import regrets
from pandas.core.groupby import DataFrameGroupBy
from plotly.graph_objects import Figure

from .averagerank import AverageRank, rankings
from .averageresult import AverageResult, regrets
from .warm_start_efficiency import build_frame
import numpy as np

WarmStartExperimentsTuple = Tuple[ExperimentClient, ExperimentClient, ExperimentClient]


def _create_results_df(
    target_task: BaseTask,
    source_tasks: List[BaseTask],
    cold_start_experiments_per_task: List[List[ExperimentClient]],
    warm_start_experiments_per_task: List[List[ExperimentClient]],
    hot_start_experiments_per_task: List[List[ExperimentClient]],
    task_similarities: List[float] = None,
    with_evc_tree: bool = False,
) -> pd.DataFrame:
    import orion.analysis
    import pandas as pd

    task_similarities = task_similarities or [
        source_task.similarity(target_task) for source_task in source_tasks
    ]
    # # TODO: Debugging
    # task_similarities = [
    #     source_task.similarity(target_task) for source_task in source_tasks
    # ]
    task_strings = [str(source_task) for source_task in source_tasks]

    def make_stacked_dataframe(
        experiments_per_task: List[List[ExperimentClient]],
    ) -> pd.DataFrame:
        dataframes: Dict[int, pd.DataFrame] = {
            task_id: build_frame(experiments, with_evc_tree=with_evc_tree)
            for task_id, experiments in enumerate(experiments_per_task)
        }
        stacked_dataframe = pd.concat(dataframes, names=["task_id"])
        assert stacked_dataframe.index.names == [
            "task_id",
            "experiment_id",
            "trial_index",
        ]
        return stacked_dataframe

    cold_start_df = make_stacked_dataframe(cold_start_experiments_per_task)
    warm_start_df = make_stacked_dataframe(warm_start_experiments_per_task)
    hot_start_df = make_stacked_dataframe(hot_start_experiments_per_task)

    def _check_experiment_ids_dont_overlap(
        cold_start_df: pd.DataFrame,
        warm_start_df: pd.DataFrame,
        hot_start_df: pd.DataFrame,
    ) -> None:
        # Double-check that there is no overlap in the experiment ids between the
        # cold/warm/hot groups.
        cold_experiment_ids = set(v[1] for v in cold_start_df.index)
        warm_experiment_ids = set(v[1] for v in warm_start_df.index)
        hot_experiment_ids = set(v[1] for v in hot_start_df.index)

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
        names=["warm_start_type", "task_id", "experiment_id", "trial_index"],
    )
    # df["task_correlation"] = -1.0 * np.ones(len(df))
    # TODO:
    df: pd.DataFrame
    task_ids = df.index.get_level_values("task_id")
    task_correlations: List[float] = - np.ones(len(df))
    for i, task_correlation_factor in enumerate(task_similarities):
        task_correlations[task_ids == i] = task_correlation_factor
    df["task_correlation"] = task_correlations
    # for task_index, task_dataframe in dataframes.items():
    #     [task_similarities[task_index]] * len(task_dataframe)

    assert df.index.names == [
        "warm_start_type",
        "task_id",
        "experiment_id",
        "trial_index",
    ]
    return df


def warm_start_task_correlation_figure(
    df: pd.DataFrame = None,
    target_task: BaseTask = None,
    source_tasks: List[BaseTask] = None,
    task_similarities: Sequence[float] = None,
    cold_start_experiments_per_task: List[List[ExperimentClient]] = None,
    warm_start_experiments_per_task: List[List[ExperimentClient]] = None,
    hot_start_experiments_per_task: List[List[ExperimentClient]] = None,
    with_evc_tree: bool = True,
    algorithm_name: str = None,
) -> List[plotly.graph_objects.Figure]:
    """ TODO: Given the source task and the list of target tasks, create a figure that
    plots the 'warm start efficiency' (what that means exactly is still TBD) vs the
    correlation coefficient.
    """

    # TODO: Add random search as a line between the hot and cold, since it should also
    # improve simply because of having more points.

    # import dash
    # import dash_core_components as dcc
    # import dash_html_components as html
    # from dash.dependencies import Input, Output
    # import plotly.express as px
    # import plotly.express as px
    # df = px.data.tips()
    # df = TODO
    import orion.analysis
    import pandas as pd
    if df is None:
        df = _create_results_df(
            target_task=target_task,
            source_tasks=source_tasks,
            cold_start_experiments_per_task=cold_start_experiments_per_task,
            warm_start_experiments_per_task=warm_start_experiments_per_task,
            hot_start_experiments_per_task=hot_start_experiments_per_task,
            task_similarities=task_similarities,
            with_evc_tree=with_evc_tree,
        )
    assert df.index.names == [
        "warm_start_type",
        "task_id",
        "experiment_id",  # TODO: Change to `run_id`.
        "trial_index",
    ]
    # df.drop(columns=["order", "experiment_id"], inplace=True)

    # 'hack' to keep the multi-index, but group by the experiment id.
    # NOTE: There is probably a built-in way to do this.
    # mean_per_experiment = df[df.groupby(level="experiment_id").mean().astype(bool)]

    # NOTE: This *almost* works, but it removes the 'warm-start-type' column, since it's
    # not numeric. (Using `mean(numeric_only=False)`` doesnt work either).
    # mean_per_experiment = df.groupby(level=["warm_start_type", "experiment_id"]).mean()

    huge_df = df
    # Each resulting value is the 'average' of the values for one experiment.
    # Useful for showing the distribution of the average results of each experiment.
    grouped_by_experiment = huge_df.groupby(
        level=["task_id", "warm_start_type", "experiment_id"]
    )
    # Useful for showing the average across all trials across all experiments.
    grouped_across_trials = huge_df.groupby(level=["task_id", "warm_start_type"])

    df_grouped_by_experiment = grouped_by_experiment.describe()

    y_mean = df_grouped_by_experiment[("objective", "mean")]
    y_std = df_grouped_by_experiment[("objective", "std")]
    assert y_mean.index.names == [
        "task_id",
        "warm_start_type",
        "experiment_id",
    ]
    color_dict = {"cold": "blue", "warm": "orange", "hot": "red"}

    # average_per_task_per_type["warm_start_type"] = [
    #     v[1] for v in average_per_task_per_type.index
    # ]

    import plotly
    import plotly.express as px
    import plotly.graph_objects as go

    warm_start_types = np.unique(huge_df.index.get_level_values("warm_start_type"))
    task_ids = np.unique(huge_df.index.get_level_values("task_id"))

    index_tuples: List[Tuple[str, str, int]] = []
    n_runs: List[int] = []
    n_trials: List[int] = []
    for warm_start_type in warm_start_types:
        for task_id in task_ids:
            # index_array = huge_df.index.to_frame()
            index_array = np.array([v[:2] for v in huge_df.index])

            matching_rows_mask = (
                index_array == np.array([warm_start_type, task_id])
            ).all(1)
            matching_rows = huge_df[matching_rows_mask]
            experiment_names = matching_rows["name"]
            # NOTE: Could also get the number of trials per experiment that way:
            # unique_experiment_names, trials_per_experiment = np.unique(experiment_names, return_counts=True)
            unique_experiment_names = np.unique(experiment_names)
            n_runs.append(len(unique_experiment_names))
            n_trials.append(len(matching_rows))
            # n_runs[warm_start_type, task_id] = len(unique_experiment_names)
            index_tuples.append((warm_start_type, task_id))

    # Create the plot dataframe:
    plot_df_index = pd.MultiIndex.from_tuples(
        index_tuples, names=["warm_start_type", "task_id"]
    )
    plot_df = pd.DataFrame(index=plot_df_index)

    n_runs_series = pd.Series(data=n_runs, index=plot_df_index, name="n_runs")
    n_trials_series = pd.Series(data=n_trials, index=plot_df_index, name="n_trials")

    # groupby across the groups of experiments ([3 x n_tasks x repetitions])
    group_across_experiments = df_grouped_by_experiment.groupby(
        level=["warm_start_type", "task_id"]
    )

    plot_df["n_runs"] = n_runs_series
    plot_df["n_trials"] = n_trials_series
    task_similarities_df = group_across_experiments[
        [("task_correlation", "mean")]
    ].mean()
    plot_df["similarity(source_task, target_task)"] = task_similarities_df
    # NOTE: These don't work because of the incompatible index.
    # plot_df["mean_objective"] = grouped_by_experiment["objective"].mean()
    # plot_df["min_objective"] = grouped_by_experiment["objective"].min()
    # plot_df["mean_best"] = grouped_by_experiment["best"].mean()

    # NOTE: Only take the stats we care about.
    experiment_stats_df = df_grouped_by_experiment.groupby(
        level=["warm_start_type", "task_id"]
    )[[("objective", "mean"), ("objective", "min"), ("best", "mean")]].describe()

    for (
        experiment_field,
        stat_name,
        stat_stat_name,
    ) in experiment_stats_df.columns.to_flat_index():
        new_column_name = (
            f"{stat_name}(experiments.{experiment_field})_{stat_stat_name}"
        )
        new_column = experiment_stats_df[(experiment_field, stat_name, stat_stat_name)]
        plot_df[new_column_name] = new_column

    assert plot_df.index.names == ["warm_start_type", "task_id"], plot_df.index.names

    # NOTE: Need to add a column to enable richer plots below:
    # Both of these work, but I think the second option is easier to undestand.
    # plot_df = pd.concat([plot_df, plot_df.index.to_frame()], axis="columns")
    plot_df["task_id"] = plot_df.index.get_level_values("task_id")
    plot_df["warm_start_type"] = plot_df.index.get_level_values("warm_start_type")
    if source_tasks:
        plot_df["source_task"] = [str(source_tasks[task_id]) for task_id in plot_df["task_id"]]
    plot_df["target_task"] = [str(target_task) for _ in range(len(plot_df))]

    figures = []
    for column_name in [
        "mean(experiments.objective)",
        "min(experiments.objective)",
        "mean(experiments.best)",
    ]:
        from plotly.graph_objects import Figure

        fig: Figure = px.line(
            plot_df,
            x="similarity(source_task, target_task)",
            y=f"{column_name}_mean",
            error_y=f"{column_name}_std",
            color="warm_start_type",
            hover_data=[
                "n_runs",
                "n_trials",
                f"{column_name}_min",
                f"{column_name}_max",
                f"{column_name}_std",
                # "source_task",
                "target_task",
            ],
            color_discrete_map=color_dict,
            title=(
                f"Warm start efficiency vs task correlation: "
                f"{algorithm_name} - {str(target_task)} - {column_name}"
            ),
            # TODO: Figure out a way to show the source tasks on the x axis in addition to the similarity score.
            labels={
                f"{column_name}_min": "min",
                f"{column_name}_max": "max",
                f"{column_name}_std": "std",
                
                # round(task_similarity.item(), 2): f"{task_similarity} - {task}" for task_similarity, task
                # in zip(task_similarities_df.values, source_tasks)
            }
        )
        fig.update_layout(
             font=dict(
                # family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
            )
        )

        figures.append(fig)
    return figures
