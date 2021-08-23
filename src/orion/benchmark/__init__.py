#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark definition
======================
"""
import copy
import itertools

from tabulate import tabulate

from orion.client import create_experiment


class Benchmark:
    """
    Benchmark definition

    Parameters
    ----------
    name: str
        Name of the benchmark
    algorithms: list, optional
        Algorithms used for benchmark, and for each algorithm, it can be formats as below:

        - A `str` of the algorithm name
        - A `dict`, with only one key and one value, where key is the algorithm name and value is a dict for the algorithm config.
        - A `dict`, with two keys.

            algorithm: str or dict
                Algorithm name in string or a dict with algorithm configure.
            deterministic: bool, optional
                True if it is a deterministic algorithm, then for each assessment, only one experiment
                will be run for this algorithm.

        Examples:

        >>> ["random", "tpe"]
        >>> ["random", {"tpe": {"seed": 1}}]
        >>> [{"algorithm": "random"}, {"algorithm": {"gridsearch": {"n_values": 50}}, "deterministic": True}]

    targets: list, optional
        Targets for the benchmark, each target will be a dict with two keys.

        assess: list
            Assessment objects
        task: list
            Task objects

    storage: dict, optional
        Configuration of the storage backend.
    """

    def __init__(self, name, algorithms, targets, storage=None):
        self._id = None
        self.name = name
        self.algorithms = algorithms
        self.targets = targets
        self.metadata = {}
        self.storage_config = storage

        self.studies = []

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
                study.setup_experiments()
                self.studies.append(study)

    def process(self, n_workers=1):
        """Run studies experiment"""
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

    def analysis(self):
        """Return all the assessment figures"""
        figures = []
        for study in self.studies:
            figure = study.analysis()
            figures.append(figure)
        return figures

    def experiments(self, silent=True):
        """Return all the experiments submitted in benchmark"""
        experiment_table = []
        for study in self.studies:
            for exp in study.experiments():
                exp_column = dict()
                stats = exp.stats
                exp_column["Algorithm"] = list(exp.configuration["algorithms"].keys())[
                    0
                ]
                exp_column["Experiment Name"] = exp.name
                exp_column["Number Trial"] = len(exp.fetch_trials())
                exp_column["Best Evaluation"] = stats["best_evaluation"]
                experiment_table.append(exp_column)

        if not silent:
            print("Total Experiments: {}".format(len(experiment_table)))
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
            deterministic = False

            if isinstance(algorithm, dict):
                if len(algorithm) > 1 or algorithm.get("algorithm"):
                    deterministic = algorithm.get("deterministic", False)
                    experiment_algorithm = algorithm["algorithm"]

                    if isinstance(experiment_algorithm, dict):
                        name, parameters = list(experiment_algorithm.items())[0]
                    else:
                        name = experiment_algorithm
                else:
                    name, parameters = list(algorithm.items())[0]
            else:
                name = algorithm

            self.algo_name = name
            self.parameters = parameters
            self.deterministic = deterministic

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

    def _build_benchmark_algorithms(self, algorithms):
        benchmark_algorithms = list()
        for algorithm in algorithms:
            benchmark_algorithm = self._StudyAlgorithm(algorithm)
            benchmark_algorithms.append(benchmark_algorithm)
        return benchmark_algorithms

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        max_trials = self.task.max_trials
        task_num = self.assessment.task_num
        space = self.task.get_search_space()

        for task_index in range(task_num):

            for algo_index, algorithm in enumerate(self.algorithms):

                # Run only 1 experiment for deterministic algorithm
                if algorithm.is_deterministic and task_index > 0:
                    continue

                experiment_name = (
                    self.benchmark.name
                    + "_"
                    + self.assess_name
                    + "_"
                    + self.task_name
                    + "_"
                    + str(task_index)
                    + "_"
                    + str(algo_index)
                )

                experiment = create_experiment(
                    experiment_name,
                    space=space,
                    algorithms=algorithm.experiment_algorithm,
                    max_trials=max_trials,
                    storage=self.benchmark.storage_config,
                )
                self.experiments_info.append((task_index, experiment))

    def execute(self, n_workers=1):
        """Execute all the experiments of the study"""
        max_trials = self.task.max_trials

        for _, experiment in self.experiments_info:
            # TODO: it is a blocking call
            experiment.workon(self.task, n_workers=n_workers, max_trials=max_trials)

    def status(self):
        """Return status of the study"""
        algorithm_tasks = {}

        for _, experiment in self.experiments_info:
            trials = experiment.fetch_trials()

            algorithm_name = list(experiment.configuration["algorithms"].keys())[0]

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

    def analysis(self):
        """Return assessment figure"""
        return self.assessment.analysis(self.task_name, self.experiments_info)

    def experiments(self):
        """Return all the experiments of the study"""
        exps = []
        for _, experiment in self.experiments_info:
            exps.append(experiment)
        return exps

    def __repr__(self):
        """Represent the object as a string."""
        algorithms_list = [algorithm.name for algorithm in self.algorithms]

        return "Study(assessment=%s, task=%s, algorithms=[%s])" % (
            self.assess_name,
            self.task_name,
            ",".join(algorithms_list),
        )
