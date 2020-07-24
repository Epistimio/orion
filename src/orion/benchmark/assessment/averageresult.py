import importlib
from orion.benchmark.base import BaseAssess
from tabulate import tabulate
from orion.benchmark import Benchmark


class AverageResult(BaseAssess):

    def __init__(self, algorithms, task, benchmark, average_num=10):
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

        self.task_num = average_num

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

        for task_index in range(self.task_num):
            for algo_index, algorithm in enumerate(self.algorithms):

                if isinstance(algorithm, dict):
                    algorithm_name = algorithm.keys()[0]
                else:
                    algorithm_name = algorithm

                task_name = task_prefix + '_' + self.__class__.__name__ + \
                            '_' + self.task_class.__name__ + '_' + \
                            str(task_index) + '_' + str(algo_index);
                task_inst = self.task_class(task_name, algorithm, assess=self)
                task_inst.run()

                self.tasks.append((algorithm_name, task_index, task_inst))

    def status(self):
        """
        - get the overall status of the assess, like how many tasks to run and the status of each task(experiment)
        [
          {
            'algorithm': 'random',
            'assessment': 'AverageResult',
            'completed': 1,
            'experiments': 1,
            'task': 'RosenBrock',
            'trials': 10
          },
          {
            'algorithm': 'tpe',
            'assessment': 'AverageResult',
            'completed': 1,
            'experiments': 1,
            'task': 'RosenBrock',
            'trials': 10
          }
        ]
        :return:
        """
        algorithm_tasks = {}
        for task_info in self.tasks:

            algorithm_name, task_index, task = task_info
            state = task.status()
            if algorithm_tasks.get(algorithm_name, None) is None:
                task_state = {'algorithm': state['algorithm'], 'experiments': 0,
                              'assessment': self.__class__.__name__, 'task': self.task_class.__name__,
                              'completed': 0, 'trials' : 0}
            else:
                task_state = algorithm_tasks[algorithm_name]

            task_state['experiments'] = task_state['experiments'] + len(state['experiments'])

            is_done = 0
            trials_num = 0
            for exp in state['experiments']:
                if exp['is_done']:
                    is_done += 1
                trials_num += sum([len(value) for value in exp['trials'].values()])

            task_state['trials'] = task_state['trials'] + trials_num
            task_state['completed'] = task_state['completed'] + is_done

            algorithm_tasks[algorithm_name] = task_state

        assess_status = list(algorithm_tasks.values())

        return assess_status

    def result(self):
        """
        -  json format of the result
        :return:
        """
        pass

    def display(self, notebook=False):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """

        algorithm_tasks = {}
        best_evals = {}
        for algo_index, algorithm in enumerate(self.algorithms):

            if isinstance(algorithm, dict):
                algorithm_name = algorithm.keys()[0]
            else:
                algorithm_name = algorithm

            task_state = {'Algorithm': algorithm_name,
                          'Assessment': self.__class__.__name__, 'Task': self.task_class.__name__}
            algorithm_tasks[algorithm_name] = task_state
            best_evals[algorithm_name] = []

        for task_info in self.tasks:

            algorithm_name, task_index, task = task_info

            experiments = task.performance()
            for exp in experiments:
                stats = exp.stats
                best_evals[algorithm_name].append(stats['best_evaluation'])

        for algo, evals in best_evals.items():
            evals.sort()
            best = evals[0]
            average = sum(evals)/len(evals)
            algorithm_tasks[algo]['Average Evaluation'] = average
            algorithm_tasks[algo]['Best Evaluation'] = best
            algorithm_tasks[algo]['Experiments Number'] = len(evals)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(list(algorithm_tasks.values()), headers='keys', tablefmt='html', stralign='center', numalign='center')))
        else:
            table = tabulate(list(algorithm_tasks.values()), headers='keys', tablefmt='grid', stralign='center', numalign='center')
            print(table)

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass
