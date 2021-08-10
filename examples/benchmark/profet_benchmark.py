import logging
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import List, Type

import numpy as np
import pandas as pd
import plotly
from simple_parsing.helpers import choice

# from warmstart.new_knowledge_base import KnowledgeBase
from warmstart.tasks.profet import SvmTask
from warmstart.tasks.quadratics import QuadraticsTask
from warmstart.tasks.task import Task

from orion.benchmark.assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
)
from orion.benchmark.warm_start_benchmark import WarmStartTaskCorrelationBenchmark

# NOTE: Hot-start case current has the source points with different task id, but same
# objective function as the target task. We can't set the task id to 0 in the source
# points, otherwise not enough points will get sampled from the algo.
from orion.benchmark.benchmark_client import get_or_create_benchmark

from orion.benchmark.assessment import AverageResult, AverageRank

from orion.benchmark.task import RosenBrock, EggHolder, CarromTable
from orion.benchmark.task.quadratics import QuadraticsTask
from orion.benchmark.task.profet import SvmTask, FcNetTask, ForresterTask, XgBoostTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig

import logging
from logging import getLogger as get_logger

# orion_logger = get_logger("orion")
# orion_logger.setLevel(logging.ERROR)

logger = get_logger("orion.benchmark.task")
logger.setLevel(logging.INFO)

from orion.benchmark.task.profet import XgBoostTask, FcNetTask, SvmTask, ForresterTask


# BUG: #629 (https://github.com/Epistimio/orion/issues/629)
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing.space import build_space
_ = PrimaryAlgo(space=build_space(), algorithm_config="random") 


@dataclass
class ProfetExperimentConfig:
    """ Configuration option for the demo of the Profet tasks. """
    # The type of Profet task to create.
    task_type: Type[Task] = choice(
        {
            "svm": SvmTask,
            "fcnet": FcNetTask,
            "xgboost": XgBoostTask,
            "forrester": ForresterTask,
        }
    )

    # Name of the experiment.
    name: str = "profet"
    # Configuration options for the training of the meta-model used in the Profet tasks.
    profet_train_config: MetaModelTrainingConfig = MetaModelTrainingConfig()
    
    # Number of repetitions for each experiment
    n_repetitions: int = 3
    # Optimization budget (max number of trials) for optimizing the task.
    max_trials: int = 25
    # Run in debug mode:
    # - No presistent storage
    # - More verbose logging
    debug: bool = False
    # Random seed.
    # TODO: Not currently passed to the Benchmark, only used to create the target task
    # (and the source task too I think?)
    seed: int = 123
    # Path to the pickledb file to use for the `storage` argument of the benchmark.
    storage_pickle_path: Path = Path("profet.pkl")
    # Path to the folder where figures are to be saved.
    figures_dir: Path = Path("figures")


    def __post_init__(self):
        simple_parsing_logger = get_logger("simple_parsing")
        if not self.debug:
            simple_parsing_logger.setLevel(logging.WARNING)


def main(config: ProfetExperimentConfig):
    task = config.task_type(
        task_id=0,
        max_trials=config.max_trials,
        seed=config.seed,
        train_config=config.profet_train_config,
    )

    benchmark = get_or_create_benchmark(
        name=config.name,
        algorithms=["robo_gp", "robo_ablr", "robo_dngo"],
        targets=[
            {
                "assess": [AverageResult(config.n_repetitions)],
                "task": [
                    task,
                    # QuadraticsTask(25),
                    # RosenBrock(25, dim=3),
                    # EggHolder(20, dim=4),
                    # CarromTable(20),
                    # SvmTask(max_trials=config.max_trials, train_config=train_config),
                    # FcNetTask(max_trials=30, train_config=train_config),
                    # ForresterTask(max_trials=30, train_config=train_config),
                    # XgBoostTask(max_trials=30, train_config=train_config),
                ],
            }
        ],
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(config.storage_pickle_path)},
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
        figure.write_html(
            str(figures_dir / f"warm_start_fig_{i:02}.html"), include_plotlyjs="cdn"
        )


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser(__doc__)
    parser.add_arguments(ProfetExperimentConfig, "config")
    args = parser.parse_args()
    config: ProfetExperimentConfig = args.config
    main(config)
