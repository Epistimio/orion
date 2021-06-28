# TODO: Not sure if this is totally necessary, but the idea is that we need to swap out
# the type of Study to use.

import itertools
import copy
from typing import Dict, List, Type, Union

from orion.benchmark.assessment.base import BaseAssess
from orion.benchmark.task.base import BaseTask
from orion.core.worker.multi_task_algo import AbstractKnowledgeBase

from .assessment.warm_start_efficiency import WarmStartEfficiency
from .assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
    _create_results_df,
)
from .benchmark import Benchmark
from .study import Study
from .warm_start_study import WarmStartStudy

from typing_extensions import TypedDict, runtime_checkable, Protocol



class TargetsDict(TypedDict):
    assess: List[BaseAssess]
    task: List[BaseTask]


class WarmStartTargetsDict(TypedDict):
    assess: List[WarmStartEfficiency]
    source_tasks: List[Union[BaseTask, List[BaseTask]]]
    target_tasks: List[BaseTask]


class WarmStartBenchmark(Benchmark):
    def __init__(
        self,
        name: str,
        algorithms: List[Union[str, Dict[str, Union[str, Dict]]]],
        source_tasks: List[Union[BaseTask, List[BaseTask]]],
        target_tasks: List[BaseTask],
        knowledge_base_type: Type[AbstractKnowledgeBase],
        repetitions: int = 5,
        debug: bool = False,
        storage: Dict = None,
    ):
        super().__init__(name, algorithms, targets=[], storage=storage)
        self.knowledge_base_type = knowledge_base_type
        self.source_tasks: List[List[BaseTask]] = [
            source_task if isinstance(source_task, list) else [source_task]
            for source_task in source_tasks
        ]
        self.target_tasks: List[BaseTask] = target_tasks if isinstance(
            target_tasks, list
        ) else [target_tasks]
        self.repetitions = repetitions
        self.debug = debug
        # Dict mapping from algorithm name to a list of WarmStartStudies, one for each
        # (source_task(s), target_task) pair.
        # self.studies_dict: Dict[str, List[WarmStartStudy]] = {}
        # self.setup_studies()

    def setup_studies(self):
        """Setup studies to run for the benchmark.
        Benchmark `algorithms`, together with each `task` and `assessment` combination
        define a study.
        """
        assert not self.studies
        for index, (source_tasks, target_task) in enumerate(
            zip(self.source_tasks, self.target_tasks)
        ):
            assessment = WarmStartEfficiency(self.repetitions)
            study = WarmStartStudy(
                benchmark=self,  # TODO: remove `benchmark` arg to Study.
                algorithms=self.algorithms,
                assessment=assessment,
                source_tasks=source_tasks,
                target_task=target_task,
                knowledge_base_type=self.knowledge_base_type,
                warm_start_seed=123,  # TODO: Vary this?
                # target_task_index=index,
                debug=self.debug,
            )
            study.setup_experiments()
            # if algorithm_name not in self.studies:
            #     self.studies_dict[algorithm_name] = []
            # self.studies_dict[algorithm_name].append(study)
            self.studies.append(study)

    def process(self, n_workers=1):
        """Run studies experiment"""
        for study in self.studies:
            study.execute(n_workers)

    @property
    def configuration(self) -> Dict:
        """Return a copy of an `Benchmark` configuration as a dictionary."""
        config = dict()

        config["name"] = self.name
        config["algorithms"] = self.algorithms

        config["source_tasks"] = [
            [task.configuration for task in source_tasks]
            for source_tasks in self.source_tasks
        ]
        config["target_tasks"] = [
            [task.configuration for task in source_tasks]
            for source_tasks in self.source_tasks
        ]
        config["repetitions"] = self.repetitions
        config["storage"] = self.storage_config

        if self.id is not None:
            config["_id"] = self.id

        return copy.deepcopy(config)

    def analysis(self):
        """Return all the assessment figures"""
        figures = []
        # TODO: Figure out a better place to call this! (Need to create a figure using
        # the results of multiple studies).
        assert all(isinstance(study, WarmStartStudy) for study in self.studies)

        if all(
            self.target_tasks[0] == target_task and len(source_tasks) == 1
            for source_tasks, target_task in zip(self.source_tasks, self.target_tasks)
        ):
            # WIP: If all the target tasks are the same, then add a figure that compares
            # the warm-start efficiency vs the correlation between the source and target
            # tasks.

            target_task = self.target_tasks[0]
            source_tasks = self.source_tasks
            assert all(
                isinstance(source_tasks, BaseTask) or len(source_tasks) == 1
                for source_tasks in self.source_tasks
            ), (
                "can only create this figure if there is only one source task per "
                "target task for now."
            )
            # Replace length-1 lists with their first item.
            # TODO: For now (june 2) this is always just QuadraticsTasks.
            source_tasks: List[BaseTask] = [
                source_task_group[0] for source_task_group in self.source_tasks
            ]

            assert len(source_tasks) == len(self.source_tasks) == len(self.studies), (len(source_tasks), len(self.source_tasks), len(self.studies), self.studies)

            # Re-order the keys of the multi-level dictionary:
            # {
            #     algo name -> {
            #         task index -> [
            #             list of [list of experiments
            #                      (of len `self.repetitions`)]
            #             (of len `len(self.target_tasks)`)]
            #     }
            # }

            cold_start_experiments: Dict[
                str, List[List[ExperimentClient]]
            ] = defaultdict(list)
            warm_start_experiments: Dict[
                str, List[List[ExperimentClient]]
            ] = defaultdict(list)
            hot_start_experiments: Dict[
                str, List[List[ExperimentClient]]
            ] = defaultdict(list)

            for task_index, study in enumerate(self.studies):
                for algo_index, algo_name in enumerate(study.algorithms):
                    cold_start_experiments[algo_name].append(
                        study.cold_start_experiments[algo_index]
                    )
                    warm_start_experiments[algo_name].append(
                        study.warm_start_experiments[algo_index]
                    )
                    hot_start_experiments[algo_name].append(
                        study.hot_start_experiments[algo_index]
                    )

            assert set(cold_start_experiments.keys()) == set(self.algorithms)
            assert set(warm_start_experiments.keys()) == set(self.algorithms)
            assert set(hot_start_experiments.keys()) == set(self.algorithms)
            result_dfs: Dict[str, pd.DataFrame] = {}
            for algo_name in self.algorithms:
                # DEBUGGING: Why are the tasks always the same?
                # assert False, cold_start_experiments[algo_name]
                algo_results_df = _create_results_df(
                    target_task=target_task,
                    source_tasks=source_tasks,
                    cold_start_experiments_per_task=cold_start_experiments[algo_name],
                    warm_start_experiments_per_task=warm_start_experiments[algo_name],
                    hot_start_experiments_per_task=hot_start_experiments[algo_name],
                )
                result_dfs[algo_name] = algo_results_df
                task_correlation_figure = warm_start_task_correlation_figure(
                    df=algo_results_df, algorithm_name=algo_name,
                )
                figures.append(task_correlation_figure)

            # self.results_df = pd.concat(result_dfs, names=["algorithm"])
            # return figures

        for study in self.studies:
            figure = study.analysis()
            if isinstance(figure, list):
                figures.extend(figure)
            else:
                figures.append(figure)
        return figures
