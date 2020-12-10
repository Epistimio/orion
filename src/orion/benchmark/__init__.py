#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark` -- Benchmark definition
================================================================

.. module:: benchmark
   :platform: Unix
   :synopsis: Benchmark definition.

"""
import copy
import itertools

from tabulate import tabulate

from orion.client import create_experiment


class Benchmark():
    """
    Benchmark definition

    Parameters
    ----------
    name: str
        Name of the benchmark
    algorithms: list, optional
        Algorithms used for benchmark, each algorithm can be a string or dict.
    targets: list, optional
        Targets for the benchmark, each target will be a dict with two keys.

        assess: list
            Assessment objects
        task: list
            Task objects
    """

    def __init__(self, name, algorithms, targets):
        self._id = None
        self.name = name
        self.algorithms = algorithms
        self.targets = targets
        self.metadata = {}

        self.studies = []

    def setup_studies(self):
        """Setup studies to run for the benchmark.
        Benchmark `algorithms`, together with each `task` and `assessment` combination
        define a study.
        """
        for target in self.targets:
            assessments = target['assess']
            tasks = target['task']

            for assess, task in (itertools.product(*[assessments, tasks])):
                study = Study(self, self.algorithms, assess, task)
                study.setup_experiments()
                self.studies.append(study)

    def process(self):
        """Run studies experiment"""
        for study in self.studies:
            study.execute()

    def status(self, silent=False, notebook=True):
        """Display benchmark status"""
        total_exp_num = 0
        complete_exp_num = 0
        total_trials = 0
        benchmark_status = []
        for study in self.studies:
            for status in study.status():
                column = dict()
                column['Algorithms'] = status['algorithm']
                column['Assessments'] = status['assessment']
                column['Tasks'] = status['task']
                column['Total Experiments'] = status['experiments']
                total_exp_num += status['experiments']
                column['Completed Experiments'] = status['completed']
                complete_exp_num += status['completed']
                column['Submitted Trials'] = status['trials']
                total_trials += status['trials']
                benchmark_status.append(column)

        if not silent:
            print('Benchmark name: {}, Experiments: {}/{}, Submitted trials: {}'.
                  format(self.name, complete_exp_num, total_exp_num, total_trials))
            if notebook:
                from IPython.display import HTML, display
                display(HTML(tabulate(benchmark_status, headers='keys',
                                      tablefmt='html', stralign='center', numalign='center')))
            else:
                table = tabulate(benchmark_status, headers='keys',
                                 tablefmt='grid', stralign='center', numalign='center')
                print(table)

        return benchmark_status

    def analysis(self, notebook=True):
        """Display benchmark result"""
        for study in self.studies:
            figure = study.display(notebook)
            figure.show()

    def experiments(self, silent=False, notebook=True):
        """Return all the experiments submitted in benchmark"""
        experiment_table = []
        for study in self.studies:
            for exp in study.experiments():
                exp_column = dict()
                stats = exp.stats
                exp_column['Algorithm'] = list(exp.configuration['algorithms'].keys())[0]
                exp_column['Experiment Name'] = exp.name
                exp_column['Number Trial'] = len(exp.fetch_trials())
                exp_column['Best Evaluation'] = stats['best_evaluation']
                experiment_table.append(exp_column)

        if not silent:
            if notebook:
                from IPython.display import HTML, display
                display(HTML(tabulate(experiment_table, headers='keys',
                                      tablefmt='html', stralign='center', numalign='center')))
            else:
                table = tabulate(experiment_table, headers='keys',
                                 tablefmt='grid', stralign='center', numalign='center')
                print(table)
            print('Total Experiments: {}'.format(len(experiment_table)))
        return experiment_table

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the experiment in the database if configured.

        Value is `None` if the experiment is not configured.
        """
        return self._id

    @property
    def configuration(self):
        """Return a copy of an `Benchmark` configuration as a dictionary."""
        config = dict()

        config['name'] = self.name
        config['algorithms'] = self.algorithms

        targets = []
        for target in self.targets:
            str_target = {}
            assessments = target['assess']
            str_assessments = []
            for assessment in assessments:
                str_assessments.append(assessment.configuration)
            str_target['assess'] = str_assessments

            tasks = target['task']
            str_tasks = []
            for task in tasks:
                str_tasks.append(task.configuration)
            str_target['task'] = str_tasks
        targets.append(str_target)
        config['targets'] = targets

        if self.id is not None:
            config['_id'] = self.id

        # print('config:', config)
        return copy.deepcopy(config)


class Study():
    """
    Benchmark definition

    Parameters
    ----------
    benchmark: `Benchmark` instance
    algorithms: list
        Algorithms used for benchmark, each algorithm can be a string or dict.
    assessment: list
        `Assessment` instance
    task: list
        `Task` instance
    """

    def __init__(self, benchmark, algorithms, assessment, task):
        self.algorithms = algorithms
        self.assessment = assessment
        self.task = task
        self.benchmark = benchmark

        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.experiments_info = []

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        max_trials = self.task.max_trials
        task_num = self.assessment.task_num
        space = self.task.get_search_space()

        for task_index in range(task_num):

            for algo_index, algorithm in enumerate(self.algorithms):

                experiment_name = self.benchmark.name + '_' + self.assess_name + \
                            '_' + self.task_name + '_' + \
                            str(task_index) + '_' + str(algo_index);
                experiment = create_experiment(
                    experiment_name, space=space, algorithms=algorithm,
                    max_trials=max_trials)
                self.experiments_info.append((task_index, experiment))

    def execute(self):
        """Execute all the experiments of the study"""
        blackbox_fun = self.task.get_blackbox_function()
        max_trials = self.task.max_trials

        for _, experiment in self.experiments_info:
            # TODO: it is a blocking call
            experiment.workon(blackbox_fun, max_trials)

    def status(self):
        """Return status of the study"""
        algorithm_tasks = {}

        for _, experiment in self.experiments_info:
            trials = experiment.fetch_trials()

            algorithm_name = list(experiment.configuration['algorithms'].keys())[0]

            if algorithm_tasks.get(algorithm_name, None) is None:
                task_state = {'algorithm': algorithm_name, 'experiments': 0,
                              'assessment': self.assess_name, 'task': self.task_name,
                              'completed': 0, 'trials': 0}
            else:
                task_state = algorithm_tasks[algorithm_name]

            task_state['experiments'] += 1
            task_state['trials'] += len(trials)
            if experiment.is_done:
                task_state['completed'] += 1

            algorithm_tasks[algorithm_name] = task_state

        return list(algorithm_tasks.values())

    def display(self, notebook=True):
        """Return assessment figure"""
        return self.assessment.plot_figures(self.task_name, self.experiments_info)

    def experiments(self):
        """Return all the experiments of the study"""
        exps = []
        for _, experiment in self.experiments_info:
            exps.append(experiment)
        return exps

    def __repr__(self):
        """Represent the object as a string."""
        algorithms_list = list()
        for algorithm in self.algorithms:
            if isinstance(algorithm, dict):
                algorithm_name = list(algorithm.keys())[0]
            else:
                algorithm_name = algorithm
            algorithms_list.append(algorithm_name)

        return "Study(assessment=%s, task=%s, algorithms=[%s])" % \
            (self.assess_name, self.task_name, ','.join(algorithms_list))
