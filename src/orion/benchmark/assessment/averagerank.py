#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.assessment` -- Average Rank Assessment
================================================================

.. module:: assessment
   :platform: Unix
   :synopsis: Benchmark algorithms with average rank.

"""

from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.express as px


from orion.benchmark.base import BaseAssess

class AverageRank(BaseAssess):
    """
    For each algorithm, run fixed number of Experiment, average the rank of trials for
    the same algorithm at the same trial sequence order.
    For the performance of trials in an Experiment, instead using the actual trial objective value,
    here we use the best objective value in the same Experiment until the particular trial.
    """

    def __init__(self, task_num=1):
        super(AverageRank, self).__init__(task_num=task_num)

    def plot_figures(self, task, experiments):

        task_algorithm_exp = defaultdict(list)

        for task_index, exp in experiments:
            algorithm_name = list(exp.configuration['algorithms'].keys())[0]

            trials = list(filter(lambda trial: trial.status == 'completed', exp.fetch_trials()))
            exp_trails = self._build_exp_trails(trials)

            task_algorithm_exp[task_index].append((algorithm_name, exp_trails))

        ploty = self._display_plot(task, task_algorithm_exp)

        return ploty

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
            for trial_index, argsorts in enumerate(algo_sorted_trials):
                for argsort_index, algo_index in enumerate(argsorts):
                    algo_ranks[trial_index][algo_index] = argsort_index + 1
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

        return fig

    def _build_exp_trails(self, trials):
        """
        1. sort the trials wrt. submit time
        2. reset the objective value of each trail with the best until it
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
