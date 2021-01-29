#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.assessment`."""

import copy

import plotly
import pytest

import orion.core.io.experiment_builder as experiment_builder
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.testing import OrionState, create_experiment


class TestAverageRank:
    """Test assessment AverageRank"""

    def test_creation(self):
        """Test creation"""
        ar1 = AverageRank()
        assert ar1.task_num == 1
        assert ar1.configuration == {"AverageRank": {"task_num": 1}}

        ar2 = AverageRank(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {"AverageRank": {"task_num": 5}}

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

    @pytest.mark.usefixtures("version_XYZ")
    def test_figure_layout(
        self, benchmark_algorithms, generate_experiment_trials, task_number
    ):
        """Test assessment plot format"""
        ar1 = AverageRank()
        algo_num = len(benchmark_algorithms)

        gen_exps, gen_trials = generate_experiment_trials

        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            for i in range(task_number * algo_num):
                experiment = experiment_builder.build("experiment-name-{}".format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.analysis("task_name", experiments)

        assert type(plot) is plotly.graph_objects.Figure

        assert plot.layout.title.text == "Average Rankings"
        assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
        assert plot.layout.yaxis.title.text == "Ranking based on loss"

        assert len(plot.data) == len(benchmark_algorithms) * 2

        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
        for i in range(algo_num):
            algo_name = next(iter(benchmark_algorithms[i].keys()))

            line_trace = line_plots[i]
            assert line_trace.type == "scatter"
            assert line_trace.name == algo_name
            assert line_trace.mode == "lines"
            assert len(line_trace.y) == 3
            assert len(line_trace.x) == 3

            err_trace = err_plots[i]
            assert err_trace.fill == "toself"
            assert err_trace.name == algo_name
            assert not err_trace.showlegend
            # * 2 because trace is above and under the mean
            assert len(err_trace.y) == 3 * 2
            assert len(err_trace.x) == 3 * 2


class TestAverageResult:
    """Test assessment AverageResult"""

    def test_creation(self):
        """Test creation"""
        ar1 = AverageResult()
        assert ar1.task_num == 1
        assert ar1.configuration == {"AverageResult": {"task_num": 1}}

        ar2 = AverageResult(task_num=5)
        assert ar2.task_num == 5
        assert ar2.configuration == {"AverageResult": {"task_num": 5}}

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

    @pytest.mark.usefixtures("version_XYZ")
    def test_figure_layout(
        self, benchmark_algorithms, generate_experiment_trials, task_number
    ):
        """Test assessment plot format"""
        ar1 = AverageResult()
        algo_num = len(benchmark_algorithms)

        gen_exps, gen_trials = generate_experiment_trials
        experiments = list()
        with OrionState(experiments=gen_exps, trials=gen_trials):
            for i in range(task_number * algo_num):
                experiment = experiment_builder.build("experiment-name-{}".format(i))
                experiments.append((i, copy.deepcopy(experiment)))
            plot = ar1.analysis("task_name", experiments)

        assert type(plot) is plotly.graph_objects.Figure

        assert plot.layout.title.text == "Average Regret"
        assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
        assert plot.layout.yaxis.title.text == "loss"

        assert len(plot.data) == len(benchmark_algorithms) * 2

        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
        for i in range(algo_num):
            algo_name = next(iter(benchmark_algorithms[i].keys()))

            line_trace = line_plots[i]
            assert line_trace.type == "scatter"
            assert line_trace.name == algo_name
            assert line_trace.mode == "lines"
            assert len(line_trace.y) == 3
            assert len(line_trace.x) == 3

            err_trace = err_plots[i]
            assert err_trace.fill == "toself"
            assert err_trace.name == algo_name
            assert not err_trace.showlegend
            # * 2 because trace is above and under the mean
            assert len(err_trace.y) == 3 * 2
            assert len(err_trace.x) == 3 * 2
