#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.assessment` -- Average Rank Result
================================================================

.. module:: assessment
   :platform: Unix
   :synopsis: Benchmark algorithms with average result.

"""
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px

from orion.benchmark.base import BaseAssess


class AverageResult(BaseAssess):
    """
    For each algorithm, run fixed number of Experiment, average the performance of trials
    for the same algorithm at the same trial sequence order.
    For the performance of trials in an Experiment, instead using the actual trial objective
    value, here we use the best objective value in the same Experiment until the particular trial.
    """

    def __init__(self, task_num=1):
        super(AverageResult, self).__init__(task_num=task_num)

    def plot_figures(self, task, experiments):
        """
        Generate a `plotly.graph_objects.Figure`

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to run for
            this assessment, and experiment is an instance of `orion.core.worker.experiment`.
        """
        algorithm_exp_trials = defaultdict(list)

        for _, exp in experiments:
            algorithm_name = list(exp.configuration["algorithms"].keys())[0]

            trials = list(
                filter(lambda trial: trial.status == "completed", exp.fetch_trials())
            )
            exp_trails = self._build_exp_trails(trials)
            algorithm_exp_trials[algorithm_name].append(exp_trails)

        ploty = self._display_plot(task, algorithm_exp_trials)
        return ploty

    def _display_plot(self, task, algorithm_exp_trials):

        algorithm_averaged_trials = {}
        plot_tables = []
        for algo, sorted_trails in algorithm_exp_trials.items():
            data = np.array(sorted_trails).transpose().mean(axis=-1)
            algorithm_averaged_trials[algo] = data
            df = pd.DataFrame(data, columns=["objective"])
            df["algorithm"] = algo
            plot_tables.append(df)

        df = pd.concat(plot_tables)
        title = "Assessment {} over Task {}".format(self.__class__.__name__, task)
        fig = px.line(
            df,
            y="objective",
            labels={"index": "trial_seq"},
            color="algorithm",
            title=title,
        )
        return fig

    def _build_exp_trails(self, trials):
        """
        1. sort the trials wrt. submit time
        2. reset the objective value of each trail with the best until it
        """
        data = [[trial.submit_time, trial.objective.value] for trial in trials]
        sorted(data, key=lambda x: x[0])

        result = []
        smallest = np.inf
        for _, objective in enumerate(data):
            if smallest > objective[1]:
                smallest = objective[1]
                result.append(objective[1])
            else:
                result.append(smallest)
        return result
