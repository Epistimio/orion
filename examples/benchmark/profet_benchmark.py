""" Example script for using tasks from the Profet algorithm.

This can be called like so:
```console
python examples/benchmark/profet_benchmark.py --task_type svm
```
"""
import logging
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import List, Type

from orion.algo.base import algo_factory
from orion.benchmark.assessment import AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task.base import BenchmarkTask
from orion.benchmark.task.profet import (
    ProfetFcNetTask,
    ProfetForresterTask,
    ProfetSvmTask,
    ProfetXgBoostTask,
)
from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from orion.storage.base import setup_storage

try:
    from simple_parsing.helpers import choice
except ImportError as exc:
    raise RuntimeError(
        "Need simple-parsing to be installed to run this example.\n"
        "You can install it using `pip install simple-parsing`."
    ) from exc

logger = get_logger("orion")
logger.setLevel(logging.DEBUG)


algos_available = set(algo_factory.get_classes().keys())


@dataclass
class ProfetExperimentConfig:
    """Configuration option for the demo of the Profet tasks."""

    # The type of Profet task to create.
    task_type: Type[BenchmarkTask] = choice(  # type: ignore
        {
            "svm": ProfetSvmTask,
            "fcnet": ProfetFcNetTask,
            "xgboost": ProfetXgBoostTask,
            "forrester": ProfetForresterTask,
        }
    )

    # Name of the experiment.
    name: str = "profet"
    # Configuration options for the training of the meta-model used in the Profet tasks.
    profet_train_config: MetaModelConfig = MetaModelConfig()

    algorithms: List[str] = choice(
        *algos_available, default_factory=["random", "tpe"].copy
    )

    # Number of repetitions for each experiment
    n_repetitions: int = 10
    # Optimization budget (max number of trials) for optimizing the task.
    max_trials: int = 50
    # Run in debug mode:
    # - No persistent storage
    # - More verbose logging
    # (@TODO: This isn't technically correct: Benchmarks don't support the `debug` flag in
    # `develop` but that feature is to be added by the long-standing warm-start PR.
    debug: bool = False
    # Random seed.
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
            model_config=config.profet_train_config,
            input_dir=config.input_dir,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
    else:
        # NOTE: This doesn't normally happen when using this from the command-line.
        task = config.task_type(max_trials=config.max_trials)

    print(f"Storage file used: {config.storage_pickle_path}")

    storage = setup_storage(
        {
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(config.storage_pickle_path)},
        }
    )

    benchmark = get_or_create_benchmark(
        storage,
        name=config.name,
        algorithms=config.algorithms,
        targets=[
            {
                "assess": [AverageResult(config.n_repetitions)],
                "task": [
                    task,
                ],
            }
        ],
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
