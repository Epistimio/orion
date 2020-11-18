import importlib
from collections import defaultdict

from orion.benchmark.base import BaseAssess
from tabulate import tabulate
import pandas as pd
import plotly.express as px
import numpy as np


class AverageRank(BaseAssess):
    """
    For each algorithm, run fixed number of Experiment, average the rank of trials for the same algorithm
    at the same trial sequence order.
    For the performance of trials in an Experiment, instead using the actual trial objective value, here we use the
    best objective value in the same Experiment until the particular trial.
    """

    def __init__(self, task_num=1):
        super(AverageRank, self).__init__(task_num=task_num)

    def display(self, task, experiments, notebook=True):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        best_evals = defaultdict(list)
        task_algorithm_exp = defaultdict(list)

        for task_index, exp in experiments:
            stats = exp.stats
            algorithm_name = list(exp.configuration['algorithms'].keys())[0]

            best_evals[algorithm_name].append(stats['best_evaluation'])

            trials = list(filter(lambda trial: trial.status == 'completed', exp.fetch_trials()))
            exp_trails = self._build_exp_trails(trials)

            task_algorithm_exp[task_index].append((algorithm_name, exp_trails))

        self._display_table(task, best_evals, notebook)
        self._display_plot(task, task_algorithm_exp)

    def _display_table(self, task,  best_evals, notebook):

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

    def _display_plot(self, task, task_algorithm_exp):

        algorithm_trials_ranks = defaultdict(list)
        for index, algo_exp_trials in task_algorithm_exp.items():

            index_trials = []
            index_algo = []
            for algo, exp_trials in algo_exp_trials:
                index_algo.append(algo)
                index_trials.append(exp_trials)

            # [n_algo, n_trial] => [n_trial, n_algo],
            # then argsort the trial objective at different timestamp
            algo_sorted_trials = np.array(index_trials).transpose().argsort()

            # replace the sort index for each trail among different algorithms
            algo_ranks = np.zeros(algo_sorted_trials.shape, dtype=int)
            for trail_index, argsorts in enumerate(algo_sorted_trials):
                for argsort_index, algo_index in enumerate(argsorts):
                    algo_ranks[trail_index][algo_index] = argsort_index + 1
            # [n_trial, n_algo] => [n_algo, n_trial]
            algo_ranks = algo_ranks.transpose()

            for algo_index, ranks in enumerate(algo_ranks):
                algorithm_trials_ranks[index_algo[algo_index]].append(ranks)

        plot_tables = []
        for algo, ranks in algorithm_trials_ranks.items():
            data = np.array(ranks).transpose().mean(axis=-1)
            df = pd.DataFrame(data, columns=['rank'])
            df['algorithm'] = algo
            plot_tables.append(df)

        df = pd.concat(plot_tables)
        title = 'Assessment {} over Task {}'.format(self.__class__.__name__, task)
        fig = px.line(df, y='rank', labels={'index': 'trial_seq'}, color='algorithm', title=title)
        fig.show()

    def _build_exp_trails(self, trials):
        """
        1. sort the trials wrt. submit time
        2. reset the objective value of each trail with the best until it
        :param trials:
        :return:
        """
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
