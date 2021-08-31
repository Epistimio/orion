import logging
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import List, Type

import numpy as np
import pandas as pd
import plotly
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import BaseTask, CarromTable, EggHolder, RosenBrock
from orion.benchmark.task.profet import FcNetTask, ForresterTask, SvmTask, XgBoostTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask
from orion.benchmark.task.quadratics import QuadraticsTask
from simple_parsing.helpers import choice, list_field

logger = get_logger("orion.benchmark.task")
logger.setLevel(logging.INFO)

from orion.benchmark.task.profet import FcNetTask, ForresterTask, SvmTask, XgBoostTask

# BUG: #629 (https://github.com/Epistimio/orion/issues/629)
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing.space import build_space

_ = PrimaryAlgo(space=build_space(), algorithm_config="random")


@dataclass
class ProfetExperimentConfig:
    """ Configuration option for the demo of the Profet tasks. """

    # The type of Profet task to create.
    task_type: Type[BaseTask] = choice(
        {"svm": SvmTask, "fcnet": FcNetTask, "xgboost": XgBoostTask, "forrester": ForresterTask,}
    )

    # Name of the experiment.
    name: str = "profet"
    # Configuration options for the training of the meta-model used in the Profet tasks.
    profet_train_config: MetaModelTrainingConfig = MetaModelTrainingConfig()

    algorithms: List[str] = list_field("robo_gp", "robo_ablr", "robo_dngo")

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

    input_dir: Path = Path("profet_data")
    checkpoint_dir: Path = Path("profet_outputs")

    def __post_init__(self):
        simple_parsing_logger = get_logger("simple_parsing")
        if not self.debug:
            simple_parsing_logger.setLevel(logging.WARNING)


def main(config: ProfetExperimentConfig):
    if issubclass(config.task_type, ProfetTask):
        task = config.task_type(
            max_trials=config.max_trials,
            task_id=0,
            train_config=config.profet_train_config,
            input_dir=config.input_dir,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
    else:
        task = config.task_type(max_trials=config.max_trials)

    benchmark = get_or_create_benchmark(
        name=config.name,
        algorithms=config.algorithms,
        targets=[{"assess": [AverageResult(config.n_repetitions)], "task": [task,],}],
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

    benchmark.process(n_workers=1)

    status = benchmark.status(False)
    print(status)
    # if all(isinstance(source_task, QuadraticsTask) for source_task in benchmark.source_tasks):
    figures = benchmark.analysis()
    # # # TODO: Instead of having like 30 figures, should try and create an interactive-ish
    # # # plotly thingy that can switch between the different quantities.
    # import plotly.io as pio
    # pio.renderers.default = "browser"
    for i, figure in enumerate(figures):
        figure.show()
        figure.write_image(str(figures_dir / f"fig_{i:02}.svg"))
        figure.write_image(str(figures_dir / f"fig_{i:02}.png"))
        figure.write_html(str(figures_dir / f"fig_{i:02}.html"), include_plotlyjs="cdn")


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(__doc__)
    parser.add_arguments(ProfetExperimentConfig, "config")
    args = parser.parse_args()
    config: ProfetExperimentConfig = args.config
    main(config)
