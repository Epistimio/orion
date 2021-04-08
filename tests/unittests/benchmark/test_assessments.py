#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.assessment`."""

import plotly
import pytest

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.testing import create_experiment, create_study_experiments
from orion.testing.plotting import assert_rankings_plot, assert_regrets_plot


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
    def test_figure_layout(self, study_experiments_config):
        """Test assessment plot format"""
        ar1 = AverageRank()

        with create_study_experiments(**study_experiments_config) as experiments:
            plot = ar1.analysis("task_name", experiments)

            assert_rankings_plot(
                plot,
                [
                    list(algorithm["algorithm"].keys())[0]
                    for algorithm in study_experiments_config["algorithms"]
                ],
                balanced=study_experiments_config["max_trial"],
                with_avg=True,
            )


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

    def test_analysis(self, experiment_config, trial_config):
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
    def test_figure_layout(self, study_experiments_config):
        """Test assessment plot format"""
        ar1 = AverageResult()

        with create_study_experiments(**study_experiments_config) as experiments:
            plot = ar1.analysis("task_name", experiments)

            assert_regrets_plot(
                plot,
                [
                    list(algorithm["algorithm"].keys())[0]
                    for algorithm in study_experiments_config["algorithms"]
                ],
                balanced=study_experiments_config["max_trial"],
                with_avg=True,
            )
