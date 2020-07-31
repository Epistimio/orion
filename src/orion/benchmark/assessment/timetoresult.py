import importlib
from orion.benchmark.base import BaseAssess
from tabulate import tabulate
from orion.benchmark import Benchmark
import pandas as pd
import plotly.graph_objects as go


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

    def display(self, notebook=False):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        time_to_result = []
        algorithm_exp_trials = {}
        for task in self.tasks:
            experiments = task.performance()
            for exp in experiments:
                exp_column = dict()
                stats = exp.stats
                trial_id = stats['best_trials_id']
                best_trial = exp.get_trial(uid=trial_id)
                exp_column['Algorithm'] = list(exp.configuration['algorithms'].keys())[0]
                exp_column['Assessment'] = self.__class__.__name__
                exp_column['Experiment Name'] = exp.name
                exp_column['Best Trial'] = trial_id  # TODO use sequential id
                exp_column['Best Evaluation'] = stats['best_evaluation']
                for param_name, param_value in best_trial.params.items():
                    exp_column[param_name] = param_value
                time_to_result.append(exp_column)

                algo = list(exp.configuration['algorithms'].keys())[0]
                trials = list(filter(lambda trial: trial.status == 'completed', exp.fetch_trials()))

                algorithm_exp_trials[algo] = self._build_frame(trials, algo)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(time_to_result, headers='keys', tablefmt='html', stralign='center', numalign='center')))
        else:
            table = tabulate(time_to_result, headers='keys', tablefmt='grid', stralign='center', numalign='center')
            print(table)

        fig = go.Figure()
        for algo, df in algorithm_exp_trials.items():
            fig.add_scatter(y=df['objective'],
                            mode='lines',
                            name=algo)
        title = 'Assessment {} over Task {}'.format(self.__class__.__name__, self.task_class.__name__)
        fig.update_layout(title=title,
                          xaxis_title="seq",
                          yaxis_title='objective')
        fig.show()

    def _build_frame(self, trials, algorithm, order_by='suggested'):
        """Builds the dataframe for the plot"""
        data = [(trial.status,
                trial.submit_time,
                trial.objective.value) for trial in trials]

        df = pd.DataFrame(data, columns=['status', 'suggested', 'objective'])

        df = df.sort_values(order_by)

        del df['status']
        del df['suggested']

        df.index.name = 'seq'
        return df

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass
