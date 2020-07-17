import itertools
import importlib


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
        for assessment in self.assessments:
            print(assessment.status())

    def analysis(self):
        """
        - display the result of each assess
        :return:
        """
        pass

    def register(self):
        """
        register benchmark object into db
        :return:
        """
        pass
