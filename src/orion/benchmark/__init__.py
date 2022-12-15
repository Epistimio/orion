#!/usr/bin/env python
"""
Benchmark definition
======================
"""
import copy
import itertools
import time
from collections import defaultdict

from tabulate import tabulate

import orion.core
from orion.algo.base import algo_factory
from orion.client import create_experiment
from orion.executor.base import executor_factory
from orion.storage.base import BaseStorageProtocol


class Benchmark:
    """
    Benchmark definition

    Parameters
    ----------
    storage: Storage
        Instance of the storage to use

    name: str
        Name of the benchmark

    algorithms: list, optional
        Algorithms used for benchmark, and for each algorithm, it can be formats as below:

        - A `str` of the algorithm name
        - A `dict`, with only one key and one value, where key is the algorithm name and value is a dict for the algorithm config.

        Examples:

        >>> ["random", "tpe"]
        >>> ["random", {"tpe": {"seed": 1}}]

    targets: list, optional
        Targets for the benchmark, each target will be a dict with two keys.

        assess: list
            Assessment objects
        task: list
            Task objects

    executor: `orion.executor.base.BaseExecutor`, optional
        Executor to run the benchmark experiments
    """

    def __init__(
        self,
        storage,
        name,
        algorithms,
        targets,
        executor=None,
    ):
        assert isinstance(storage, BaseStorageProtocol)

        self._id = None
        self.name = name
        self.algorithms = algorithms
        self.targets = targets
        self.metadata = {}
        self.storage = storage
        self._executor = executor
        self._executor_owner = False

        self.studies = []

    @property
    def executor(self):
        """Returns the current executor to use to run jobs in parallel"""
        if self._executor is None:
            self._executor_owner = True
            self._executor = executor_factory.create(
                orion.core.config.worker.executor,
                n_workers=orion.core.config.worker.n_workers,
                **orion.core.config.worker.executor_configuration,
            )

        return self._executor

    def setup_studies(self):
        """Setup studies to run for the benchmark.
        Benchmark `algorithms`, together with each `task` and `assessment` combination
        define a study.
        """
        for target in self.targets:
            assessments = target["assess"]
            tasks = target["task"]

            for assess, task in itertools.product(*[assessments, tasks]):
                study = Study(self, self.algorithms, assess, task)
                self.studies.append(study)

    def process(self, n_workers=1):
        """Run studies experiment"""
        if self._executor is None or self._executor_owner:
            # TODO: Do the experiments really use the executor set here??
            with self.executor:
                for study in self.studies:
                    study.execute(n_workers)
        else:
            for study in self.studies:
                study.execute(n_workers)

    def status(self, silent=True):
        """Display benchmark status"""
        total_exp_num = 0
        complete_exp_num = 0
        total_trials = 0
        benchmark_status = []
        for study in self.studies:
            for status in study.status():
                column = dict()
                column["Algorithms"] = status["algorithm"]
                column["Assessments"] = status["assessment"]
                column["Tasks"] = status["task"]
                column["Total Experiments"] = status["experiments"]
                total_exp_num += status["experiments"]
                column["Completed Experiments"] = status["completed"]
                complete_exp_num += status["completed"]
                column["Submitted Trials"] = status["trials"]
                total_trials += status["trials"]
                benchmark_status.append(column)

        if not silent:
            print(
                "Benchmark name: {}, Experiments: {}/{}, Submitted trials: {}".format(
                    self.name, complete_exp_num, total_exp_num, total_trials
                )
            )
            self._pretty_table(benchmark_status)

        return benchmark_status

    def analysis(self, assessment=None, task=None, algorithms=None):
        """Return all assessment figures

        Parameters
        ----------
        assessment: str or None, optional
            Filter analysis and only return those for the given assessment name.
        task: str or None, optional
            Filter analysis and only return those for the given task name.
        algorithms: list of str or None, optional
            Compute analysis only on specified algorithms. Compute on all otherwise.
        """
        self.validate_assessment(assessment)
        self.validate_task(task)
        self.validate_algorithms(algorithms)

        figures = defaultdict(dict)
        for study in self.studies:
            if (
                assessment is not None
                and study.assess_name != assessment
                or task is not None
                and study.task_name != task
            ):
                continue

            # NOTE: From ParallelAssessment PR
            # figures[study.assess_name].update(figure[study.assess_name])
            figures[study.assess_name][study.task_name] = study.analysis(algorithms)
        return figures

    def validate_assessment(self, assessment):
        if assessment is None:
            return
        assessment_names = {study.assess_name for study in self.studies}
        if assessment not in assessment_names:
            raise ValueError(
                f"Invalid assessment name: {assessment}. "
                f"It should be one of {sorted(assessment_names)}"
            )

    def validate_task(self, task):
        if task is None:
            return
        task_names = {study.task_name for study in self.studies}
        if task not in task_names:
            raise ValueError(
                f"Invalid task name: {task}. It should be one of {sorted(task_names)}"
            )

    def validate_algorithms(self, algorithms):
        if algorithms is None:
            return
        algorithm_names = {
            algo if isinstance(algo, str) else next(iter(algo.keys()))
            for algo in self.algorithms
        }

        for algorithm in algorithms:
            if algorithm not in algorithm_names:
                raise ValueError(
                    f"Invalid algorithm: {algorithm}. "
                    f"It should be one of {sorted(algorithm_names)}"
                )

    def get_experiments(self, silent=True):
        """Return all the experiments submitted in benchmark"""
        experiment_table = []
        for study in self.studies:
            for _, exp in study.get_experiments():
                exp_column = dict()
                stats = exp.stats
                exp_column["Algorithm"] = list(exp.configuration["algorithm"].keys())[0]
                exp_column["Experiment Name"] = exp.name
                exp_column["Number Trial"] = len(exp.fetch_trials())
                exp_column["Best Evaluation"] = stats.best_evaluation
                experiment_table.append(exp_column)

        if not silent:
            print(f"Total Experiments: {len(experiment_table)}")
            self._pretty_table(experiment_table)

        return experiment_table

    def _pretty_table(self, dict_list):
        """
        Print a list of same format dict as pretty table with IPython disaply(notebook) or tablute
        :param dict_list: a list of dict where each dict has the same keys.
        """
        try:
            from IPython.display import HTML, display

            display(
                HTML(
                    tabulate(
                        dict_list,
                        headers="keys",
                        tablefmt="html",
                        stralign="center",
                        numalign="center",
                    )
                )
            )
        except ImportError:
            table = tabulate(
                dict_list,
                headers="keys",
                tablefmt="grid",
                stralign="center",
                numalign="center",
            )
            print(table)

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the benchmark in the database if configured.

        Value is `None` if the benchmark is not configured.
        """
        return self._id

    @property
    def configuration(self):
        """Return a copy of an `Benchmark` configuration as a dictionary."""
        config = dict()

        config["name"] = self.name
        config["algorithms"] = self.algorithms

        targets = []
        for target in self.targets:
            str_target = {}
            assessments = target["assess"]
            str_assessments = dict()
            for assessment in assessments:
                str_assessments.update(assessment.configuration)
            str_target["assess"] = str_assessments

            tasks = target["task"]
            str_tasks = dict()
            for task in tasks:
                str_tasks.update(task.configuration)
            str_target["task"] = str_tasks

        targets.append(str_target)
        config["targets"] = targets

        if self.id is not None:
            config["_id"] = self.id

        return copy.deepcopy(config)

    def __del__(self):
        self.close()

    def close(self):
        if self._executor_owner:
            self._executor.close()


class Study:
    """
    A study is one assessment and task combination in the `Benchmark` targets. It will
    build and run experiments for all the algorithms for that task.

    Parameters
    ----------
    benchmark: A Benchmark instance
    algorithms: list
        Algorithms used for benchmark, each algorithm can be a string or dict, with same format as `Benchmark` algorithms.
    assessment: list
        `Assessment` instance
    task: list
        `Task` instance
    """

    class _StudyAlgorithm:
        """
        Represent user input json format algorithm as a Study algorithm object for easy to use.
        Parameters
        ----------
        algorithm: one algorithm in the `Study` algorithms list.
        """

        def __init__(self, algorithm):
            parameters = None

            if isinstance(algorithm, dict):
                name, parameters = list(algorithm.items())[0]
            else:
                name = algorithm

            self.algo_name = name
            self.parameters = parameters
            self.deterministic = algo_factory.get_class(name).deterministic

        @property
        def name(self):
            return self.algo_name

        @property
        def experiment_algorithm(self):
            if self.parameters:
                return {self.algo_name: self.parameters}
            else:
                return self.algo_name

        @property
        def is_deterministic(self):
            return self.deterministic

    def __init__(self, benchmark, algorithms, assessment, task):
        self.algorithms = self._build_benchmark_algorithms(algorithms)
        self.assessment = assessment
        self.task = task
        self.benchmark = benchmark

        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.experiments_info = []
        self.has_assesment_executor = bool(assessment.get_executor(0))

    def _build_benchmark_algorithms(self, algorithms):
        benchmark_algorithms = list()
        for algorithm in algorithms:
            benchmark_algorithm = self._StudyAlgorithm(algorithm)
            benchmark_algorithms.append(benchmark_algorithm)
        return benchmark_algorithms

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        max_trials = self.task.max_trials
        repetitions = self.assessment.repetitions
        space = self.task.get_search_space()

        for repetition_index in range(repetitions):

            for algo_index, algorithm in enumerate(self.algorithms):

                # Run only 1 experiment for deterministic algorithm
                if algorithm.is_deterministic and repetition_index > 0:
                    continue

                experiment_name = (
                    self.benchmark.name
                    + "_"
                    + self.assess_name
                    + "_"
                    + self.task_name
                    + "_"
                    + str(repetition_index)
                    + "_"
                    + str(algo_index)
                )

                executor = (
                    self.assessment.get_executor(repetition_index)
                    or self.benchmark.executor
                )
                experiment = create_experiment(
                    experiment_name,
                    space=space,
                    algorithm=algorithm.experiment_algorithm,
                    max_trials=max_trials,
                    storage=self.benchmark.storage,
                    executor=executor,
                )
                self.experiments_info.append((repetition_index, experiment))

    def execute(self, n_workers=1):
        """Execute all the experiments of the study"""
        max_trials = self.task.max_trials

        for _, experiment in self.get_experiments():
            # TODO: it is a blocking call
            if self.has_assesment_executor:
                experiment.workon(self.task, max_trials=max_trials)
            else:
                experiment.workon(self.task, n_workers=n_workers, max_trials=max_trials)

    def status(self):
        """Return status of the study"""
        algorithm_tasks = {}

        for _, experiment in self.get_experiments():
            trials = experiment.fetch_trials()

            algorithm_name = list(experiment.configuration["algorithm"].keys())[0]

            if algorithm_tasks.get(algorithm_name, None) is None:
                task_state = {
                    "algorithm": algorithm_name,
                    "experiments": 0,
                    "assessment": self.assess_name,
                    "task": self.task_name,
                    "completed": 0,
                    "trials": 0,
                }
            else:
                task_state = algorithm_tasks[algorithm_name]

            task_state["experiments"] += 1
            task_state["trials"] += len(trials)
            if experiment.is_done:
                task_state["completed"] += 1

            algorithm_tasks[algorithm_name] = task_state

        return list(algorithm_tasks.values())

    def analysis(self, algorithms=None):
        """Return assessment figure

        Parameters
        ----------
        algorithms: list of str or None, optional
            Compute analysis only on specified algorithms. Compute on all otherwise.
        """
        return self.assessment.analysis(
            self.task_name, self.get_experiments(algorithms)
        )

    def get_experiments(self, algorithms=None):
        """Return all the experiments of the study

        Parameters
        ----------
        algorithms: list of str or None, optional
            Return only experiments for specified algorithms. Return all otherwise.
        """
        if not self.experiments_info:
            start = time.perf_counter()
            self.setup_experiments()
        if algorithms is not None:
            algorithms = [algo_name.lower() for algo_name in algorithms]
        exps = []
        for repetition_index, experiment in self.experiments_info:
            if (
                algorithms is None
                or list(experiment.algorithm.configuration.keys())[0] in algorithms
            ):
                exps.append((repetition_index, experiment))
        return exps

    def __repr__(self):
        """Represent the object as a string."""
        algorithms_list = [algorithm.name for algorithm in self.algorithms]

        return "Study(assessment={}, task={}, algorithms=[{}])".format(
            self.assess_name,
            self.task_name,
            ",".join(algorithms_list),
        )
