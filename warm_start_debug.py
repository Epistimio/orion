from orion.benchmark.benchmark_client import get_or_create_benchmark

from orion.benchmark.assessment import AverageResult, AverageRank

from orion.benchmark.task import RosenBrock, EggHolder, CarromTable
from orion.benchmark.warm_start_study import WarmStartStudy
from orion.benchmark.assessment.warm_start_efficiency import WarmStartEfficiency
from orion.benchmark import Benchmark
from orion.benchmark.warm_start_benchmark import (
    WarmStartBenchmark,
    WarmStartTaskCorrelationBenchmark,
)

from warmstart.new_knowledge_base import KnowledgeBase
from warmstart.tasks.quadratics import QuadraticsTask
from warmstart.tasks.profet import SvmTask
from ablr.ablr import ABLR
import numpy as np
from typing import List
from pathlib import Path
import pickle
from dataclasses import dataclass

import pandas as pd
from orion.algo.robo.bohamiann import OrionBohamiannWrapper
from orion.core.worker.primary_algo import PrimaryAlgo
from warmstart.distance import similarity, distance
import plotly
from typing import Type
import orion
from orion.benchmark.assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
)

# NOTE: Hot-start case current has the source points with different task id, but same
# objective function as the target task. We can't set the task id to 0 in the source
# points, otherwise not enough points will get sampled from the algo.


@dataclass
class WarmStartExperimentConfig:
    # TODO: Add a switch to choose which task type to use for the figure.
    # task_type: Type[Task] = choice({
    #     "quadratics": QuadraticsTask
    # })

    # Number of source tasks to create.
    n_tasks: int = 4
    # Number of repetitions for each experiment
    n_repetitions: int = 3
    # Number of points to use from the source task when warm-starting on the
    # target task.
    n_source_points: int = 25
    # Optimization budget (max number of trials) for optimizing the target task.
    n_target_points: int = 25
    # Run in debug mode:
    # - No presistent storage
    # - More verbose logging
    debug: bool = False
    # Random seed.
    # TODO: Not currently passed to the Benchmark, only used to create the target task
    # (and the source task too I think?)
    seed: int = 123
    # Path to the pickledb file to use for the `storage` argument of the benchmark.
    storage_pickle_path: Path = Path("foo2.pkl")
    # Path to the folder where figures are to be saved.
    figures_dir: Path = Path("figures")


def main(config: WarmStartExperimentConfig):
    target_task = QuadraticsTask(
        a0=0.1,
        a1=0.1,
        a2=0.1,
        task_id=0,
        max_trials=config.n_source_points,
        seed=config.seed,
    )

    task_correlations = np.linspace(0, 1, num=config.n_tasks + 1, endpoint=True)

    benchmark = WarmStartTaskCorrelationBenchmark(
        name="ablr_debug",
        algorithms=[
            # "robo_bohamiann",  # Incredibly slow!
            "robo_dngo",
            "robo_gp",
            # "robo_gp_mcmc",
            # "robo_randomforest",
            "robo_ablr",
            # "robo_mtablr",
            # "random",
            # "tpe",
            # "robo_mtablr",
        ],
        task_correlations=task_correlations,
        target_task=target_task,
        knowledge_base_type=KnowledgeBase,
        n_source_points=config.n_source_points,
        repetitions=config.n_repetitions,
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": config.storage_pickle_path},
        },
        debug=config.debug,
    )
    benchmark.setup_studies()

    # Since we're using the QuadraticsTask, we can evaluate the 'similarity' between
    # them, hence we just show these figures rather than the (very large) number of other
    # potential figures.
    figures_dir = config.figures_dir / benchmark.name
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    save_path = "results_df.pkl"
    # Use this to just re-create the plots.
    if Path(save_path).exists() and False:  # FIXME: Debugging
        print(f"Loading the existing results dataframe from path: {save_path}")
        results_df: pd.DataFrame = pd.read_pickle(save_path)
        benchmark.results_df = results_df
        figures: List[plotly.graph_objects.Figure]
        for algo_name in benchmark.algorithms:
            figures = warm_start_task_correlation_figure(
                df=results_df.loc[algo_name],
                algorithm_name=algo_name,
                # task_similarities=benchmark.task_similarities,
                # target_task=target_task,
                # source_tasks=benchmark.source_tasks,
            )
            for i, fig in enumerate(figures):
                fig.show()
                fig.write_image(str(figures_dir / f"{algo_name}_task_correlation.svg"))

    else:
        benchmark.process(n_workers=1)

        status = benchmark.status(False)
        figures = benchmark.analysis()
        # for figure in figures:
        #     figure.show()

        results_df = benchmark.results_df
        print(f"Saving the results dataframe to path: {save_path}")
        results_df.to_pickle(save_path)

    # if all(isinstance(source_task, QuadraticsTask) for source_task in benchmark.source_tasks):
    figures = benchmark.analysis()
    # # # TODO: Instead of having like 30 figures, should try and create an interactive-ish
    # # # plotly thingy that can switch between the different quantities.
    # import plotly.io as pio
    # pio.renderers.default = "browser"
    for i, figure in enumerate(figures):
        figure.show()
        figure.write_image(str(figures_dir / f"warm_start_fig_{i:02}.svg"))
        figure.write_image(str(figures_dir / f"warm_start_fig_{i:02}.png"))
        figure.write_html(str(figures_dir / f"warm_start_fig_{i:02}.html"), include_plotlyjs="cdn")


if __name__ == "__main__":

    from simple_parsing import ArgumentParser

    parser = ArgumentParser(__doc__)
    parser.add_arguments(WarmStartExperimentConfig, "config")
    args = parser.parse_args()
    config: WarmStartExperimentConfig = args.config
    main(config)
