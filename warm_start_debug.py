from orion.benchmark.benchmark_client import get_or_create_benchmark

from orion.benchmark.assessment import AverageResult, AverageRank

from orion.benchmark.task import RosenBrock, EggHolder, CarromTable
from orion.benchmark.warm_start_study import WarmStartStudy
from orion.benchmark.assessment.warm_start_efficiency import WarmStartEfficiency
from orion.benchmark import Benchmark
from orion.benchmark.warm_start_benchmark import WarmStartBenchmark

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
target_task = QuadraticsTask(a0=0.1, a1=0.1, a2=0.1, task_id=0, max_trials=25)
N = 4

# BUG: Figure out why we observe '50' warm-start points, rather than 25.
from pathlib import Path
import pickle
import pandas as pd

benchmark = WarmStartBenchmark(
    name="warmstart_benchmark",
    algorithms=[
        "tpe",
    ],
    source_tasks=[
        target_task.get_similar_task(
            correlation_coefficient=i * (1 / N), task_id=i, max_trials=25
        )
        for i in range(N + 1)
    ],
    target_tasks=[target_task for _ in range(N + 1)],
    repetitions=5,
    knowledge_base_type=KnowledgeBase,
    debug=True,
)
benchmark.setup_studies()


save_path = "results_df.pkl"
if Path(save_path).exists():
    print(f"Loading the existing results dataframe from path: {save_path}")
    results_df: pd.DataFrame = pd.read_pickle(save_path)
else:
    benchmark.process()

    status = benchmark.status(False)
    figures = benchmark.analysis()
    # for figure in figures:
    #     figure.show()

    results_df = benchmark.results_df
    print(f"Saving the results dataframe to path: {save_path}")
    results_df.to_pickle(save_path)

from orion.benchmark.assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
)
for algo_name in benchmark.algorithms:
    figures = warm_start_task_correlation_figure(
        df=results_df.loc["tpe"],
        algorithm_name="tpe",
        target_task=target_task,
        source_tasks=benchmark.source_tasks,
    )
    for fig in figures:
        fig.show()

# benchmark = get_or_create_benchmark(
#     name="warmstart_benchmark",
#     algorithms=[
#         # "random",
#         "tpe",
#         # "ablr",  # Needs the RoBO wrapper
#         # "BayesianOptimizer", # Doesn't work!
#     ],
#     targets=[
#         {
#             "assess": [WarmStartEfficiency(5)],
#             "source_tasks": [
#                 target_task.get_similar_task(i * (1 / N), task_id=i, max_trials=25)
#                 for i in range(N+1)
#             ],
#             "target_tasks": [
#                 target_task for _ in range(N+1)
#             ],
#         },
#     ],
#     knowledge_base_type=KnowledgeBase,
#     debug=True,
# )


# status = benchmark.status(False)
# figures = benchmark.analysis()
# for figure in figures:
#     figure.show()

# print(benchmark.studies)
# benchmark.studies[0].analysis()
# exps = benchmark.experiments(False)
