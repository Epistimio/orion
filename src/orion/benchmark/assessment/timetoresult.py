import importlib
from orion.benchmark.base import BaseAssess


class TimeToResult(BaseAssess):

    def __init__(self, algorithms, task, benchmark):
        """
        - build assess object
        - build task object (from db[existing], or from config[new])
        """

        # TODO: task can be an object instead of package.class
        mod_str, _sep, class_str = task.rpartition('.')
        module = importlib.import_module(mod_str)
        self.task_class = getattr(module, class_str)
        self.algorithms = algorithms
        self.benchmark = benchmark

        self.tasks = []

    def execute(self):
        """
        - run the tasks
        - there may be needs to run the task multiple times (such as when assess average performance)
        :return:
        """
        for i, algorithm in enumerate(self.algorithms):
            task_name = self.benchmark.name + '_' + self.__class__.__name__ + '_' + self.task_class.__name__ + '_' + str(i);
            task_inst = self.task_class(task_name, algorithm, assess=self)
            task_inst.run()
            self.tasks.append(task_inst)

    def status(self):
        """
        - get the overall status of the assess, like how many tasks to run and the status of each task(experiment)
        :return:
        """
        assess_status = []
        for task in self.tasks:
            state = task.status()

            task_state = {'algorithm': state['algorithm'], 'experiments': len(state['experiments'])}

            is_done = 0
            trials_num = 0
            for exp in state['experiments']:
                if exp['is_done']:
                    is_done += 1
                trials_num += sum([len(value) for value in exp['trials'].values()])

            task_state['trials'] = trials_num
            task_state['completed'] = is_done

            assess_status.append(task_state)

        return assess_status

    def result(self):
        """
        -  json format of the result
        :return:
        """
        pass

    def display(self):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        pass

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass
