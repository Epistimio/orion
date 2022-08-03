#!/usr/bin/env python
"""Tests for :mod:`orion.benchmark.assessment`."""


import plotly
import pytest

from orion.benchmark.assessment import AverageRank, AverageResult, ParallelAssessment
from orion.testing import create_experiment, create_study_experiments
from orion.testing.plotting import (
    assert_durations_plot,
    assert_rankings_plot,
    assert_regrets_plot,
    asset_parallel_assessment_plot,
)


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
            figure = ar1.analysis("task_name", [(0, experiment)])

        assert (
            type(figure["AverageRank"]["task_name"]["rankings"])
            is plotly.graph_objects.Figure
        )

    @pytest.mark.usefixtures("version_XYZ")
    def test_figure_layout(self, orionstate, study_experiments_config):
        """Test assessment plot format"""
        ar1 = AverageRank()

        experiments = create_study_experiments(orionstate, **study_experiments_config)
        figure = ar1.analysis("task_name", experiments)

        assert_rankings_plot(
            figure["AverageRank"]["task_name"]["rankings"],
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
            _,
            experiment,
        ):
            figure = ar1.analysis("task_name", [(0, experiment)])

        assert (
            type(figure["AverageResult"]["task_name"]["regrets"])
            is plotly.graph_objects.Figure
        )

    @pytest.mark.usefixtures("version_XYZ")
    def test_figure_layout(self, orionstate, study_experiments_config):
        """Test assessment plot format"""
        ar1 = AverageResult()

        experiments = create_study_experiments(orionstate, **study_experiments_config)
        figure = ar1.analysis("task_name", experiments)

        assert_regrets_plot(
            figure["AverageResult"]["task_name"]["regrets"],
            [
                list(algorithm["algorithm"].keys())[0]
                for algorithm in study_experiments_config["algorithms"]
            ],
            balanced=study_experiments_config["max_trial"],
            with_avg=True,
        )


class TestParallelAssessment:
    """Test assessment ParallelAssessment"""

    def test_creation(self):
        """Test creation"""
        pa1 = ParallelAssessment()
        assert pa1.workers == [1, 2, 4]
        assert pa1.task_num == 3

        pa2 = ParallelAssessment(task_num=2)
        assert pa2.workers == [1, 1, 2, 2, 4, 4]
        assert pa2.task_num == 6

        pa3 = ParallelAssessment(executor="joblib", backend="threading")
        assert pa1.workers == [1, 2, 4]
        assert pa1.task_num == 3
        assert pa3.get_executor(0).n_workers == 1
        assert pa3.get_executor(1).n_workers == 2
        assert pa3.get_executor(2).n_workers == 4

    @pytest.mark.usefixtures("version_XYZ")
    def test_analysis(self, orionstate, study_experiments_config):
        """Test assessment plot format"""
        task_num = 2
        n_workers = [1, 2, 4]
        pa1 = ParallelAssessment(task_num=task_num, n_workers=n_workers)

        study_experiments_config["task_number"] = task_num
        study_experiments_config["n_workers"] = n_workers
        experiments = create_study_experiments(orionstate, **study_experiments_config)
        figure = pa1.analysis("task_name", experiments)

        names = []
        algorithms = []
        for algorithm in study_experiments_config["algorithms"]:
            algo = list(algorithm["algorithm"].keys())[0]
            algorithms.append(algo)

            for worker in n_workers:
                names.append(algo + "_workers_" + str(worker))

        assert len(figure["ParallelAssessment"]["task_name"]) == 3
        assert_regrets_plot(
            figure["ParallelAssessment"]["task_name"]["regrets"],
            names,
            balanced=study_experiments_config["max_trial"],
            with_avg=True,
        )

        asset_parallel_assessment_plot(
            figure["ParallelAssessment"]["task_name"]["parallel_assessment"],
            algorithms,
            3,
        )

        assert_durations_plot(
            figure["ParallelAssessment"]["task_name"]["durations"], names
        )
