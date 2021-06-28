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

# TODO: Need to make sure that in the "hot_start" case, all points have the same
# task id? Or maybe use the branching could be used?
from warmstart.distance import similarity, distance

# print(distance(task_a, task_b))
# print(similarity(task_a, task_b))

# print(task_a, task_b)
# exit()
n_source_points = 25
n_target_points = 25

target_task = QuadraticsTask(
    a0=0.1, a1=0.1, a2=0.1, task_id=0, max_trials=n_source_points, seed=123
)
N = 4

# BUG: Figure out why we observe '50' warm-start points, rather than 25.
from pathlib import Path
import pickle
import pandas as pd
from orion.algo.robo.bohamiann import OrionBohamiannWrapper
from orion.core.worker.primary_algo import PrimaryAlgo

# TODO: In order to create the 'task correlation plot', we have to run a lot more
# experiments, where some of the plots don't really matter!

task_correlations = np.linspace(0, 1, num=N + 1, endpoint=True)

benchmark = WarmStartTaskCorrelationBenchmark(
    name="warmstart_benchmark",
    algorithms=[
        "tpe",
        # {"algorithm": "bohamiann"},
        # "bohamiann",
        "robo_gp",
        # "random",  # BUG: Doesn't work.
        # Bohamiann,
    ],
    task_correlations=task_correlations,
    target_task=target_task,
    knowledge_base_type=KnowledgeBase,
    n_source_points=n_source_points,
    repetitions=3,
    storage={"type": "legacy", "database": {"type": "pickleddb", "host": "foo.pkl"},},
    debug=False,
)
benchmark.setup_studies()


# Since we're using the QuadraticsTask, we can evaluate the 'similarity' between
# them, hence we just show these figures rather than the (very large) number of other
# potential figures.
figures_dir: Path = Path("figures")
figures_dir.mkdir(exist_ok=True)
import plotly

save_path = "results_df.pkl"
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
from orion.benchmark.assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
)


figures = benchmark.analysis()
# # # TODO: Instead of having like 30 figures, should try and create an interactive-ish
# # # plotly thingy that can switch between the different quantities.
for i, figure in enumerate(figures):
    figure.show()
    figure.write_image(str(figures_dir / f"warm_start_fig_{i:02}.svg"))
    figure.write_image(str(figures_dir / f"warm_start_fig_{i:02}.png"))

