import itertools
import importlib
from tabulate import tabulate


class Benchmark():
    def __init__(self, name, algorithms, targets):
        """
        - build assess and task pairs
        """
        self.name = name
        self.algorithms = algorithms

        self.assessments = []
        for target in targets:
            assessments = target['assess']
            tasks = target['task']

            for assess, task in (itertools.product(*[assessments, tasks])):
                # TODO: assessment can be an object instead of package.class
                mod_str, _sep, class_str = assess.rpartition('.')
                module = importlib.import_module(mod_str)
                assess_class = getattr(module, class_str)

                self.assessments.append(assess_class(algorithms, task, self))

    def process(self):
        """
        - for each assess-task, build Assess object and run assess execute
        - support multiple thread based a configured thread_num ?
        :return:
        """
        for assessment in self.assessments:
            assessment.execute()

    def status(self):
        """
        - get progress/details of each assess-task
        :return:
        """
        benchmark_status = []
        for assessment in self.assessments:
            for status in assessment.status():
                column = dict()
                column['Algorithms'] = status['algorithm']
                column['Assessments'] = status['assessment']
                column['Tasks'] = status['task']
                column['Total Experiments'] = status['experiments']
                column['Completed Experiments'] = status['completed']
                column['Submitted Trials'] = status['trials']
                benchmark_status.append(column)
        table = tabulate(benchmark_status, headers='keys', tablefmt='grid')
        print(table)

    def analysis(self):
        """
        - display the result of each assess
        :return:
        """
        for assessment in self.assessments:
            assessment.display()

    def register(self):
        """
        register benchmark object into db
        :return:
        """
        pass
