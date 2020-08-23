import itertools

from tabulate import tabulate
from orion.client import create_experiment


class Benchmark():
    def __init__(self, name, algorithms, targets):
        """
        - build assess and task pairs
        """
        self.name = name
        self.algorithms = algorithms

        self.studies = []
        for target in targets:
            assessments = target['assess']
            tasks = target['task']

            for assess, task in (itertools.product(*[assessments, tasks])):
                self.studies.append(Study(self, algorithms, assess, task))

    def process(self):
        """
        - for each assess-task, build Assess object and run assess execute
        - support multiple thread based a configured thread_num ?
        :return:
        """
        for study in self.studies:
            study.execute()

    def status(self, notebook=True):
        """
        - get progress/details of each assess-task
        :return:
        """
        benchmark_status = []
        for study in self.studies:
            for status in study.status():
                column = dict()
                column['Algorithms'] = status['algorithm']
                column['Assessments'] = status['assessment']
                column['Tasks'] = status['task']
                column['Total Experiments'] = status['experiments']
                column['Completed Experiments'] = status['completed']
                column['Submitted Trials'] = status['trials']
                benchmark_status.append(column)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(benchmark_status, headers='keys', tablefmt='html', stralign='center', numalign='center')))
        else:
            table = tabulate(benchmark_status, headers='keys', tablefmt='grid', stralign='center', numalign='center')
            print(table)

    def analysis(self, notebook=True):
        """
        - display the result of each assess
        :return:
        """
        for study in self.studies:
            study.display(notebook)

    def experiments(self, notebook=True):
        experiment_table = []
        for study in self.studies:
            for exp in study.experiments():
                exp_column = dict()
                stats = exp.stats
                trial_id = stats['best_trials_id']
                exp_column['Algorithm'] = list(exp.configuration['algorithms'].keys())[0]
                exp_column['Experiment Name'] = exp.name
                exp_column['Number Trial'] = len(exp.fetch_trials())
                exp_column['Best Evaluation'] = stats['best_evaluation']
                experiment_table.append(exp_column)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(experiment_table, headers='keys', tablefmt='html', stralign='center',
                                  numalign='center')))
        else:
            table = tabulate(experiment_table, headers='keys', tablefmt='grid', stralign='center', numalign='center')
            print(table)
        print('Total Experiments: {}'.format(len(experiment_table)))

    def register(self):
        """
        register benchmark object into db
        :return:
        """
        pass


class Study():

    def __init__(self, benchmark, algorithms, assessment, task):
        self.algorithms = algorithms
        self.assessment = assessment
        self.task = task
        self.benchmark = benchmark

        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.experiments_info = []

    def execute(self):

        task_num = self.assessment.task_num
        blackbox_fun = self.task.get_task_function()
        max_trials = self.task.get_task_max_trials()
        space = self.task.get_task_space()

        for task_index in range(task_num):

            for algo_index, algorithm in enumerate(self.algorithms):

                experiment_name =  self.benchmark.name + '_' + self.assess_name + \
                            '_' + self.task_name + '_' + \
                            str(task_index) + '_' + str(algo_index);
                experiment = create_experiment(
                    experiment_name, space=space, algorithms=algorithm,
                    max_trials=max_trials)
                experiment.workon(blackbox_fun, max_trials)
                self.experiments_info.append((task_index, experiment))

    def status(self):

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
        self.assessment.display(self.task_name, self.experiments_info, notebook)

    def experiments(self):
        exps = []
        for _, experiment in self.experiments_info:
            exps.append(experiment)
        return exps
