#!/usr/bin/env python
"""Tests for :mod:`orion.benchmark.Benchmark`."""

import plotly
import pytest

import orion.core.io.experiment_builder as experiment_builder
from orion.benchmark import Benchmark, Study
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.task import CarromTable, RosenBrock
from orion.client.experiment import ExperimentClient
from orion.testing import OrionState, create_study_experiments


@pytest.fixture
def benchmark(storage, benchmark_algorithms):
    """Return a benchmark instance"""
    return Benchmark(
        storage,
        name="benchmark007",
        algorithms=benchmark_algorithms,
        targets=[
            {
                "assess": [AverageResult(2), AverageRank(2)],
                "task": [RosenBrock(25, dim=3), CarromTable(20)],
            }
        ],
    )


@pytest.fixture
def study(benchmark, benchmark_algorithms):
    """Return a study instance"""
    with benchmark.executor:
        yield Study(
            benchmark, benchmark_algorithms, AverageResult(2), RosenBrock(25, dim=3)
        )


class TestBenchmark:
    """Test Benchmark"""

    def test_creation(self, benchmark, benchmark_algorithms):
        """Test benchmark instance creation"""
        cfg = {
            "name": "benchmark007",
            "algorithms": benchmark_algorithms,
            "targets": [
                {
                    "assess": {
                        "AverageResult": {"repetitions": 2},
                        "AverageRank": {"repetitions": 2},
                    },
                    "task": {
                        "RosenBrock": {"dim": 3, "max_trials": 25},
                        "CarromTable": {"max_trials": 20},
                    },
                }
            ],
        }

        assert benchmark.configuration == cfg

    def test_setup_studies(self, benchmark):
        """Test to setup studies for benchmark"""
        with OrionState():
            benchmark.setup_studies()

            assert (
                str(benchmark.studies)
                == "[Study(assessment=AverageResult, task=RosenBrock, algorithms=[random,tpe]), "
                "Study(assessment=AverageResult, task=CarromTable, algorithms=[random,tpe]), "
                "Study(assessment=AverageRank, task=RosenBrock, algorithms=[random,tpe]), "
                "Study(assessment=AverageRank, task=CarromTable, algorithms=[random,tpe])]"
            )

        assert len(benchmark.studies) == 4

    def test_process(self, benchmark, study):
        """Test to process a benchmark"""
        study.setup_experiments()
        benchmark.studies = [study]
        benchmark.process()
        name = "benchmark007_AverageResult_RosenBrock_0_0"
        experiment = experiment_builder.build(name)
        assert experiment is not None

    @pytest.mark.usefixtures("version_XYZ")
    def test_status(
        self,
        benchmark,
        study,
        study_experiments_config,
        task_number,
        max_trial,
        orionstate,
    ):
        """Test to get the status of a benchmark"""
        experiments = create_study_experiments(orionstate, **study_experiments_config)

        study.experiments_info = list(enumerate(experiments))

        benchmark.studies = [study]

        assert benchmark.status() == [
            {
                "Algorithms": "random",
                "Assessments": "AverageResult",
                "Tasks": "RosenBrock",
                "Total Experiments": task_number,
                "Completed Experiments": task_number,
                "Submitted Trials": task_number * max_trial,
            },
            {
                "Algorithms": "tpe",
                "Assessments": "AverageResult",
                "Tasks": "RosenBrock",
                "Total Experiments": task_number,
                "Completed Experiments": task_number,
                "Submitted Trials": task_number * max_trial,
            },
        ]

    @pytest.mark.usefixtures("version_XYZ")
    def test_analysis(self, orionstate, benchmark, study, study_experiments_config):
        """Test to analysis benchmark result"""
        experiments = create_study_experiments(orionstate, **study_experiments_config)
        study.experiments_info = list(enumerate(experiments))

        benchmark.studies = [study]

        figures = benchmark.analysis()

        assert len(figures) == 1
        assert (
            type(figures[study.assess_name][study.task_name]["regrets"])
            is plotly.graph_objects.Figure
        )

    @pytest.mark.usefixtures("version_XYZ")
    def test_experiments(
        self,
        orionstate,
        benchmark,
        study,
        study_experiments_config,
        max_trial,
    ):
        """Test to get experiments list of a benchmark"""
        experiments = create_study_experiments(orionstate, **study_experiments_config)

        study.experiments_info = list(enumerate(experiments))

        benchmark.studies = [study]

        assert benchmark.get_experiments() == [
            {
                "Algorithm": "random",
                "Experiment Name": "experiment-name-0",
                "Number Trial": max_trial,
                "Best Evaluation": 0,
            },
            {
                "Algorithm": "tpe",
                "Experiment Name": "experiment-name-1",
                "Number Trial": max_trial,
                "Best Evaluation": 0,
            },
            {
                "Algorithm": "random",
                "Experiment Name": "experiment-name-2",
                "Number Trial": max_trial,
                "Best Evaluation": 0,
            },
            {
                "Algorithm": "tpe",
                "Experiment Name": "experiment-name-3",
                "Number Trial": max_trial,
                "Best Evaluation": 0,
            },
        ]


class TestStudy:
    """Test Study"""

    def test_creation(self, study):
        """Test study instance creation"""
        assert (
            str(study) == "Study(assessment=AverageResult, "
            "task=RosenBrock, algorithms=[random,tpe])"
        )

    def test_creation_algorithms(self, benchmark):
        """Test study creation with all support algorithms input format"""

        algorithms = [
            {"gridsearch": {"n_values": 1}},
            "tpe",
            {"random": {"seed": 1}},
            "asha",
        ]
        study = Study(benchmark, algorithms, AverageResult(2), RosenBrock(25, dim=3))
        assert study.algorithms[0].name == "gridsearch"
        assert study.algorithms[0].experiment_algorithm == {
            "gridsearch": {"n_values": 1}
        }
        assert study.algorithms[0].is_deterministic

        assert study.algorithms[1].name == "tpe"
        assert study.algorithms[1].experiment_algorithm == "tpe"
        assert not study.algorithms[1].is_deterministic

        assert study.algorithms[2].name == "random"
        assert study.algorithms[2].experiment_algorithm == {"random": {"seed": 1}}
        assert not study.algorithms[2].is_deterministic

        assert study.algorithms[3].name == "asha"
        assert study.algorithms[3].experiment_algorithm == "asha"
        assert not study.algorithms[3].is_deterministic

    def test_setup_experiments(self, study):
        """Test to setup experiments for study"""
        study.setup_experiments()

        assert len(study.experiments_info) == 4
        assert isinstance(study.experiments_info[0][1], ExperimentClient)

    def test_execute(self, study):
        """Test to execute a study"""
        study.setup_experiments()
        study.execute()
        name = "benchmark007_AverageResult_RosenBrock_0_0"
        experiment = experiment_builder.build(name)

        assert len(experiment.fetch_trials()) == study.task.max_trials

        assert experiment is not None

    @pytest.mark.usefixtures("version_XYZ")
    def test_status(
        self,
        orionstate,
        study,
        study_experiments_config,
        task_number,
        max_trial,
    ):
        """Test to get status of a study"""
        experiments = create_study_experiments(orionstate, **study_experiments_config)

        study.experiments_info = list(enumerate(experiments))

        assert study.status() == [
            {
                "algorithm": "random",
                "assessment": "AverageResult",
                "task": "RosenBrock",
                "experiments": task_number,
                "completed": task_number,
                "trials": task_number * max_trial,
            },
            {
                "algorithm": "tpe",
                "assessment": "AverageResult",
                "task": "RosenBrock",
                "experiments": task_number,
                "completed": task_number,
                "trials": task_number * max_trial,
            },
        ]

    @pytest.mark.usefixtures("version_XYZ")
    def test_analysis(
        self,
        orionstate,
        study,
        study_experiments_config,
    ):
        """Test to get the ploty figure of a study"""
        experiments = create_study_experiments(orionstate, **study_experiments_config)

        study.experiments_info = list(enumerate(experiments))

        figure = study.analysis()

        assert type(figure["regrets"]) is plotly.graph_objects.Figure

    def test_experiments(
        self, orionstate, study, study_experiments_config, task_number
    ):
        """Test to get experiments of a study"""
        algo_num = len(study_experiments_config["algorithms"])
        experiments = create_study_experiments(orionstate, **study_experiments_config)

        study.experiments_info = list(enumerate(experiments))

        experiments = study.get_experiments()

        assert len(experiments) == study_experiments_config["task_number"] * algo_num
        assert isinstance(experiments[0][1], ExperimentClient)
