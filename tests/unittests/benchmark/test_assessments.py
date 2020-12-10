#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.assessment`."""

import plotly

from orion.core.utils.tests import generate_trials
import orion.core.io.experiment_builder as experiment_builder

from orion.core.utils.tests import create_experiment
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.core.utils.tests import OrionState

config = dict(
    name='experiment-name',
    space={'x': 'uniform(0, 200)'},
    metadata={'user': 'test-user',
              'orion_version': 'XYZ',
              'VCS': {"type": "git",
                      "is_dirty": False,
                      "HEAD_sha": "test",
                      "active_branch": None,
                      "diff_sha": "diff"}},
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir='',
    algorithms={'random': {'seed': 1}},
    producer={'strategy': 'NoParallelStrategy'},
)

trial_config = {
    'experiment': 0,
    'status': 'completed',
    'worker': None,
    'start_time': None,
    'end_time': None,
    'heartbeat': None,
    'results': [],
    'params': []
}

algorithms=[{'random': {'seed': 1}}, {'tpe': {'seed': 1}}]


class TestAverageRank():
    def test_creation(self):

        ar1 = AverageRank()
        assert ar1.task_num == 1
        assert ar1.configuration == {'orion-benchmark-assessment-averagerank-AverageRank': {'task_num': 1}}

        ar2 = AverageRank(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {'orion-benchmark-assessment-averagerank-AverageRank': {'task_num': 5}}

    def test_plot_figures(self):
        ar1 = AverageRank()

        with create_experiment(config, trial_config, ['completed']) as (_, experiment, _):
            plot = ar1.plot_figures('task_name', [(0, experiment)])

        assert type(plot) is plotly.graph_objects.Figure

    def test_figure_layout(self):

        ar1 = AverageRank()

        gen_exps = []
        gen_trials = []
        algo_num = len(algorithms)
        for i in range(2*algo_num):
            import copy
            exp = copy.deepcopy(config)
            exp['_id'] = i
            exp['name'] = 'experiment-name-{}'.format(i)
            exp['algorithms'] = algorithms[i%algo_num]
            gen_exps.append(exp)
            for j in range(3):
                trial = copy.deepcopy(trial_config)
                trial['_id'] = '{}{}'.format(i, j)
                trial['experiment'] = i
                trials = generate_trials(trial, ['completed'])
                gen_trials.extend(trials)

        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            for i in range(2*algo_num):
                experiment = experiment_builder.build('experiment-name-{}'.format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.plot_figures('task_name', experiments)

        assert type(plot) is plotly.graph_objects.Figure

        assert plot.layout.title.text == "Assessment AverageRank over Task task_name"
        assert plot.layout.xaxis.title.text == "trial_seq"
        assert plot.layout.yaxis.title.text == "rank"

        for i in range(algo_num):
            trace1 = plot.data[i]
            assert trace1.type == "scatter"
            assert trace1.name == list(algorithms[i].keys())[0]
            assert trace1.mode == "lines"
            assert len(trace1.y) == 3
            assert len(trace1.x) == 3


class TestAverageResult():
    def test_creation(self):
        ar1 = AverageResult()
        assert ar1.task_num == 1
        assert ar1.configuration == {'orion-benchmark-assessment-averageresult-AverageResult': {'task_num': 1}}

        ar2 = AverageResult(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {'orion-benchmark-assessment-averageresult-AverageResult': {'task_num': 5}}

    def test_plot_figures(self):

        ar1 = AverageResult()

        with create_experiment(config, trial_config, ['completed']) as (_, experiment, _):
            plot = ar1.plot_figures('task_name', [(0, experiment)])

        assert type(plot) is plotly.graph_objects.Figure

    def test_figure_layout(self):
        ar1 = AverageResult()

        gen_exps = []
        gen_trials = []
        algo_num = len(algorithms)
        for i in range(2*algo_num):
            import copy
            exp = copy.deepcopy(config)
            exp['_id'] = i
            exp['name'] = 'experiment-name-{}'.format(i)
            exp['algorithms'] = algorithms[i%algo_num]
            gen_exps.append(exp)
            for j in range(3):
                trial = copy.deepcopy(trial_config)
                trial['_id'] = '{}{}'.format(i, j)
                trial['experiment'] = i
                trials = generate_trials(trial, ['completed'])
                gen_trials.extend(trials)

        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            for i in range(2*algo_num):
                experiment = experiment_builder.build('experiment-name-{}'.format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.plot_figures('task_name', experiments)

        assert type(plot) is plotly.graph_objects.Figure

        assert plot.layout.title.text == "Assessment AverageResult over Task task_name"
        assert plot.layout.xaxis.title.text == "trial_seq"
        assert plot.layout.yaxis.title.text == "objective"

        for i in range(algo_num):
            trace1 = plot.data[i]
            assert trace1.type == "scatter"
            assert trace1.name == list(algorithms[i].keys())[0]
            assert trace1.mode == "lines"
            assert len(trace1.y) == 3
            assert len(trace1.x) == 3
