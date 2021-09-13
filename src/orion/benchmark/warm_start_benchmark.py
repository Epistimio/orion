# TODO: Not sure if this is totally necessary, but the idea is that we need to swap out
# the type of Study to use.
import copy
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Type, Union

import pandas as pd
import plotly
import plotly.graph_objects as go
from orion.benchmark.assessment.base import BaseAssess
from orion.benchmark.task.base import BaseTask
from orion.client.experiment import ExperimentClient
from orion.core.worker.multi_task_algo import AbstractKnowledgeBase
from typing_extensions import TypedDict

from .assessment.warm_start_efficiency import WarmStartEfficiency
from .assessment.warm_start_task_correlation import (
    _create_results_df,
    warm_start_task_correlation_figure,
)
from .benchmark import Benchmark
from .study import Study
from .warm_start_study import WarmStartStudy


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
                target_task_index=index,
                knowledge_base_type=self.knowledge_base_type,
                warm_start_seed=123,  # TODO: Vary this?
                storage=self.storage_config,  # TODO: Maybe rename this to `storage`?
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
        for study in self.studies:
            # Figure(s)
            figure_s = study.analysis()
            if isinstance(figure_s, list):
                figures.extend(figure_s)
            else:
                figures.append(figure_s)
        return figures
