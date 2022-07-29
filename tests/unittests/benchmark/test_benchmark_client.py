#!/usr/bin/env python
"""Tests for :mod:`orion.benchmark.benchmark_client`."""
import copy
import logging

import pytest

import orion.benchmark.benchmark_client as benchmark_client
import orion.core
from orion.benchmark.assessment import AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import CarromTable, RosenBrock
from orion.client import ExperimentClient
from orion.core.utils.exceptions import NoConfigurationError
from orion.executor.joblib_backend import Joblib
from orion.storage.base import setup_storage
from orion.testing.state import OrionState


class DummyTask:
    """Dummy invalid benchmark task"""


class DummyAssess:
    """Dummy invalid benchmark assessment"""


def count_benchmarks():
    """Count experiments in storage"""
    return len(setup_storage().fetch_benchmark({}))


storage_instance = ""


class TestCreateBenchmark:
    """Test Benchmark creation"""

    def test_create_benchmark(self, benchmark_config, benchmark_config_py):
        """Test creation with valid configuration"""
        with OrionState() as cfg:
            bm1 = get_or_create_benchmark(
                cfg.storage,
                **benchmark_config_py,
            )
            bm1.close()

            bm2 = get_or_create_benchmark(cfg.storage, "bm00001")
            bm2.close()

            assert bm1.configuration == benchmark_config

            assert bm1.configuration == bm2.configuration

    def test_create_with_only_name(self):
        """Test creation with a non-existing benchmark name"""
        with OrionState() as cfg:
            name = "bm00001"
            with pytest.raises(NoConfigurationError) as exc:
                get_or_create_benchmark(cfg.storage, name).close()

            assert f"Benchmark {name} does not exist in DB" in str(exc.value)

    def test_create_with_different_configure(self, benchmark_config_py, caplog):
        """Test creation with same name but different configure"""
        with OrionState() as cfg:
            config = copy.deepcopy(benchmark_config_py)
            bm1 = get_or_create_benchmark(cfg.storage, **config)
            bm1.close()

            config = copy.deepcopy(benchmark_config_py)
            config["targets"][0]["assess"] = [AverageResult(2)]

            with caplog.at_level(
                logging.WARNING, logger="orion.benchmark.benchmark_client"
            ):
                bm2 = get_or_create_benchmark(cfg.storage, **config)
                bm2.close()

            assert bm2.configuration == bm1.configuration
            assert (
                "Benchmark with same name is found but has different configuration, "
                "which will be used for this creation." in caplog.text
            )

            caplog.clear()
            config = copy.deepcopy(benchmark_config_py)
            config["targets"][0]["task"] = [RosenBrock(26, dim=3), CarromTable(20)]
            with caplog.at_level(
                logging.WARNING, logger="orion.benchmark.benchmark_client"
            ):
                bm3 = get_or_create_benchmark(cfg.storage, **config)
                bm3.close()

            assert bm3.configuration == bm1.configuration
            assert (
                "Benchmark with same name is found but has different configuration, "
                "which will be used for this creation." in caplog.text
            )

    def test_create_with_invalid_algorithms(self, benchmark_config_py):
        """Test creation with a not existed algorithm"""
        with OrionState() as cfg:

            with pytest.raises(NotImplementedError) as exc:
                benchmark_config_py["algorithms"] = [
                    {"algorithm": {"fake_algorithm": {"seed": 1}}}
                ]
                # Pass executor to close it properly
                with Joblib(n_workers=2, backend="threading") as executor:
                    get_or_create_benchmark(
                        cfg.storage, **benchmark_config_py, executor=executor
                    )
            assert "Could not find implementation of BaseAlgorithm" in str(exc.value)

    def test_create_with_deterministic_algorithm(self, benchmark_config_py):
        algorithms = [
            {"algorithm": {"random": {"seed": 1}}},
            {"algorithm": {"gridsearch": {"n_values": 50}}, "deterministic": True},
        ]
        with OrionState() as cfg:
            config = copy.deepcopy(benchmark_config_py)
            config["algorithms"] = algorithms
            bm = get_or_create_benchmark(cfg.storage, **config)
            bm.close()

            for study in bm.studies:
                for status in study.status():
                    algo = status["algorithm"]
                    if algo == "gridsearch":
                        assert status["experiments"] == 1
                    else:
                        assert status["experiments"] == study.assessment.task_num

    def test_create_with_invalid_targets(self, benchmark_config_py):
        """Test creation with invalid Task and Assessment"""
        with OrionState() as cfg:

            with pytest.raises(AttributeError) as exc:
                config = copy.deepcopy(benchmark_config_py)
                config["targets"] = [
                    {"assess": [AverageResult(2)], "task": [DummyTask]}
                ]
                get_or_create_benchmark(cfg.storage, **config).close()

            assert "type object '{}' has no attribute ".format("DummyTask") in str(
                exc.value
            )

            with pytest.raises(AttributeError) as exc:
                config = copy.deepcopy(benchmark_config_py)
                config["targets"] = [
                    {"assess": [DummyAssess], "task": [RosenBrock(25, dim=3)]}
                ]
                get_or_create_benchmark(cfg.storage, **config).close()

            assert "type object '{}' has no attribute ".format("DummyAssess") in str(
                exc.value
            )

    def test_create_with_not_loaded_targets(self, benchmark_config):
        """Test creation with assessment or task does not exist or not loaded"""

        cfg_invalid_assess = copy.deepcopy(benchmark_config)
        cfg_invalid_assess["targets"][0]["assess"]["idontexist"] = {"task_num": 2}

        with OrionState(benchmarks=cfg_invalid_assess) as cfg:
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(
                    cfg.storage,
                    benchmark_config["name"],
                ).close()
            assert "Could not find implementation of BenchmarkAssessment" in str(
                exc.value
            )

        cfg_invalid_task = copy.deepcopy(benchmark_config)
        cfg_invalid_task["targets"][0]["task"]["idontexist"] = {"max_trials": 2}

        with OrionState(benchmarks=cfg_invalid_task) as cfg:
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(
                    cfg.storage,
                    benchmark_config["name"],
                )
            assert "Could not find implementation of BenchmarkTask" in str(exc.value)

    def test_create_with_not_exist_targets_parameters(self, benchmark_config):
        """Test creation with not existing assessment parameters"""

        benchmark_config["targets"][0]["assess"]["AverageResult"] = {
            "task_num": 2,
            "idontexist": 100,
        }

        with OrionState(benchmarks=benchmark_config) as cfg:
            with pytest.raises(TypeError) as exc:
                get_or_create_benchmark(cfg.storage, benchmark_config["name"])
            assert "__init__() got an unexpected keyword argument 'idontexist'" in str(
                exc.value
            )

    def test_create_from_db_config(self, benchmark_config):
        """Test creation from existing db configubenchmark_configre"""
        with OrionState(benchmarks=copy.deepcopy(benchmark_config)) as cfg:
            bm = get_or_create_benchmark(cfg.storage, benchmark_config["name"])
            bm.close()
            assert bm.configuration == benchmark_config

    def test_create_race_condition(
        self, benchmark_config, benchmark_config_py, monkeypatch, caplog
    ):
        """Test creation in race condition"""
        with OrionState(benchmarks=benchmark_config) as cfg:

            def insert_race_condition(*args, **kwargs):
                if insert_race_condition.count == 0:
                    data = {}
                else:
                    data = benchmark_config

                insert_race_condition.count += 1

                return data

            insert_race_condition.count = 0
            monkeypatch.setattr(
                benchmark_client, "_fetch_benchmark", insert_race_condition
            )

            with caplog.at_level(
                logging.INFO, logger="orion.benchmark.benchmark_client"
            ):
                bm = benchmark_client.get_or_create_benchmark(
                    cfg.storage, **benchmark_config_py
                )
                bm.close()

            assert (
                "Benchmark registration failed. This is likely due to a race condition. "
                "Now rolling back and re-attempting building it." in caplog.text
            )
            assert insert_race_condition.count == 2

            del benchmark_config["_id"]

            assert bm.configuration == benchmark_config
            assert count_benchmarks() == 1

    def test_create_with_executor(self, benchmark_config, benchmark_config_py):

        with OrionState() as cfg:
            config = copy.deepcopy(benchmark_config_py)
            bm1 = get_or_create_benchmark(cfg.storage, **config)
            bm1.close()

            assert bm1.configuration == benchmark_config
            assert bm1.executor.n_workers == orion.core.config.worker.n_workers

            with Joblib(n_workers=2, backend="threading") as executor:
                config["executor"] = executor
                bm2 = get_or_create_benchmark(cfg.storage, **config)

                assert bm2.configuration == benchmark_config
                assert bm2.executor.n_workers == executor.n_workers
                assert orion.core.config.worker.n_workers != 2

    def test_experiments_parallel(self, benchmark_config_py, monkeypatch):
        import multiprocessing

        class FakeFuture:
            def __init__(self, value):
                self.value = value

            def wait(self, timeout=None):
                return

            def ready(self):
                return True

            def get(self, timeout=None):
                return self.value

            def successful(self):
                return True

        count = multiprocessing.Value("i", 0)
        is_done_value = multiprocessing.Value("i", 0)

        def is_done(self):
            return count.value > 0

        def submit(*args, c=count, **kwargs):
            # because worker == 2 only 2 jobs were submitted
            # we now set is_done to True so when runner checks
            # for adding more jobs it will stop right away
            c.value += 1
            return FakeFuture([dict(name="v", type="objective", value=1)])

        with OrionState() as cfg:
            config = copy.deepcopy(benchmark_config_py)

            with Joblib(n_workers=5, backend="threading") as executor:
                monkeypatch.setattr(ExperimentClient, "is_done", property(is_done))
                monkeypatch.setattr(executor, "submit", submit)

                config["executor"] = executor
                bm1 = get_or_create_benchmark(cfg.storage, **config)
                client = bm1.studies[0].experiments_info[0][1]

                count.value = 0
                bm1.process(n_workers=2)
                assert count.value == 2
                assert executor.n_workers == 5
                assert orion.core.config.worker.n_workers != 2

                is_done.done = False
                count.value = 0
                bm1.process(n_workers=3)
                assert count.value == 3
                assert executor.n_workers == 5
                assert orion.core.config.worker.n_workers != 3
