#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.assessment`."""

import plotly

import orion.core.io.experiment_builder as experiment_builder
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.testing import OrionState, create_experiment, generate_trials


class TestAverageRank:
    """Test assessment AverageRank"""

    def test_creation(self):
        """Test creation"""
        ar1 = AverageRank()
        assert ar1.task_num == 1
        assert ar1.configuration == {
            "orion-benchmark-assessment-averagerank-AverageRank": {"task_num": 1}
        }

        ar2 = AverageRank(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {
            "orion-benchmark-assessment-averagerank-AverageRank": {"task_num": 5}
        }

    def test_analysis(self, experiment_config, trial_config):
        """Test assessment plot"""
        ar1 = AverageRank()

        with create_experiment(experiment_config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = ar1.analysis("task_name", [(0, experiment)])

        assert type(plot) is plotly.graph_objects.Figure

    def test_figure_layout(self, algorithms, generate_experiment_trials):
        """Test assessment plot format"""
        ar1 = AverageRank()
        algo_num = len(algorithms)

        gen_exps, gen_trials = generate_experiment_trials
        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            import copy

            for i in range(2 * algo_num):
                experiment = experiment_builder.build("experiment-name-{}".format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.analysis("task_name", experiments)

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


class TestAverageResult:
    """Test assessment AverageResult"""

    def test_creation(self):
        """Test creation"""
        ar1 = AverageResult()
        assert ar1.task_num == 1
        assert ar1.configuration == {
            "orion-benchmark-assessment-averageresult-AverageResult": {"task_num": 1}
        }

        ar2 = AverageResult(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {
            "orion-benchmark-assessment-averageresult-AverageResult": {"task_num": 5}
        }

    def test_plot_figures(self, experiment_config, trial_config):
        """Test assessment plot"""
        ar1 = AverageResult()

        with create_experiment(experiment_config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = ar1.analysis("task_name", [(0, experiment)])

        assert type(plot) is plotly.graph_objects.Figure

    def test_figure_layout(self, algorithms, generate_experiment_trials):
        """Test assessment plot format"""
        ar1 = AverageResult()
        algo_num = len(algorithms)

        gen_exps, gen_trials = generate_experiment_trials
        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            import copy

            for i in range(2 * algo_num):
                experiment = experiment_builder.build("experiment-name-{}".format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.analysis("task_name", experiments)

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
