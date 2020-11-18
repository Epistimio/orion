from collections import defaultdict

from orion.benchmark.base import BaseAssess
from tabulate import tabulate
import pandas as pd
import plotly.express as px
import numpy as np


class AverageResult(BaseAssess):
    """
    For each algorithm, run fixed number of Experiment, average the performance of trials for the same algorithm
    at the same trial sequence order.
    For the performance of trials in an Experiment, instead using the actual trial objective value, here we use the
    best objective value in the same Experiment until the particular trial.
    """

    def __init__(self, task_num=1):
        super(AverageResult, self).__init__(task_num=task_num)

    def display(self, task, experiments, notebook=True):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        best_evals = defaultdict(list)
        algorithm_exp_trials = defaultdict(list)

        for _, exp in experiments:
            stats = exp.stats
            algorithm_name = list(exp.configuration['algorithms'].keys())[0]

            best_evals[algorithm_name].append(stats['best_evaluation'])

            trials = list(filter(lambda trial: trial.status == 'completed', exp.fetch_trials()))
            exp_trails = self._build_exp_trails(trials)
            algorithm_exp_trials[algorithm_name].append(exp_trails)

        self._display_table(task, best_evals, notebook)
        self._display_plot(task, algorithm_exp_trials)

    def _display_plot(self, task, algorithm_exp_trials):

        algorithm_averaged_trials = {}
        plot_tables = []
        for algo, sorted_trails in algorithm_exp_trials.items():
            data = np.array(sorted_trails).transpose().mean(axis=-1)
            algorithm_averaged_trials[algo] = data
            df = pd.DataFrame(data, columns=['objective'])
            df['algorithm'] = algo
            plot_tables.append(df)

        df = pd.concat(plot_tables)
        title = 'Assessment {} over Task {}'.format(self.__class__.__name__, task)
        fig = px.line(df, y='objective', labels={'index': 'trial_seq'}, color='algorithm', title=title)
        fig.show()

    def _display_table(self, task, best_evals, notebook):

        algorithm_tasks = {}
        for algo, evals in best_evals.items():
            evals.sort()
            best = evals[0]
            average = sum(evals) / len(evals)

            algorithm_tasks[algo] = {'Assessment': self.__class__.__name__, 'Task': task}
            algorithm_tasks[algo]['Algorithm'] = algo
            algorithm_tasks[algo]['Average Evaluation'] = average
            algorithm_tasks[algo]['Best Evaluation'] = best
            algorithm_tasks[algo]['Experiments Number'] = len(evals)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(list(algorithm_tasks.values()), headers='keys', tablefmt='html', stralign='center',
                                  numalign='center')))
        else:
            table = tabulate(list(algorithm_tasks.values()), headers='keys', tablefmt='grid', stralign='center',
                             numalign='center')
            print(table)

    def _build_exp_trails(self, trials):
        data = [[trial.submit_time,
                 trial.objective.value] for trial in trials]
        sorted(data, key=lambda x: x[0])

        result = []
        smallest = np.inf
        for idx, objective in enumerate(data):
            if smallest > objective[1]:
                smallest = objective[1]
                result.append(objective[1])
            else:
                result.append(smallest)
        return result
