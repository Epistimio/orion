import importlib
from orion.benchmark.base import BaseAssess
from tabulate import tabulate
from orion.benchmark import Benchmark


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

        if isinstance(self.benchmark, Benchmark):
            task_prefix = self.benchmark.name
        else:
            task_prefix = self.benchmark

        for i, algorithm in enumerate(self.algorithms):
            task_name = task_prefix + '_' + self.__class__.__name__ + '_' + self.task_class.__name__ + '_' + str(i);
            task_inst = self.task_class(task_name, algorithm, assess=self)
            task_inst.run()
            self.tasks.append(task_inst)

    def status(self):
        """
        - get the overall status of the assess, like how many tasks to run and the status of each task(experiment)
        [
          {
            'algorithm': 'random',
            'assessment': 'TimeToResult',
            'completed': 1,
            'experiments': 1,
            'task': 'RosenBrock',
            'trials': 10
          },
          {
            'algorithm': 'tpe',
            'assessment': 'TimeToResult',
            'completed': 1,
            'experiments': 1,
            'task': 'RosenBrock',
            'trials': 10
          }
        ]
        :return:
        """
        assess_status = []
        for task in self.tasks:
            state = task.status()

            task_state = {'algorithm': state['algorithm'], 'experiments': len(state['experiments']),
                          'assessment': self.__class__.__name__, 'task': self.task_class.__name__}

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
        time_to_result = []
        for task in self.tasks:
            experiments = task.performance()
            for exp in experiments:
                exp_column = dict()
                stats = exp.stats
                trial_id = stats['best_trials_id']
                best_trial = exp.get_trial(uid=trial_id)
                exp_column['Experiment Name'] = exp.name
                exp_column['Best Trial'] = trial_id  # TODO use sequential id
                exp_column['Best Evaluation'] = stats['best_evaluation']
                for param_name, param_value in best_trial.params.items():
                    exp_column[param_name] = param_value
                time_to_result.append(exp_column)

        table = tabulate(time_to_result, headers='keys', tablefmt='grid')
        print(table)

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass
