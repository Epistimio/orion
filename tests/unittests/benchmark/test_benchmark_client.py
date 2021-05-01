#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.benchmark_client`."""
import copy
import logging

import pytest

import orion.benchmark.benchmark_client as benchmark_client
import orion.core
from orion.benchmark.assessment import AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import CarromTable, RosenBrock
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.utils.singleton import SingletonNotInstantiatedError, update_singletons
from orion.storage.base import get_storage
from orion.storage.legacy import Legacy
from orion.testing.state import OrionState


class DummyTask:
    """Dummy invalid benchmark task"""

    pass


class DummyAssess:
    """Dummy invalid benchmark assessment"""

    pass


def count_benchmarks():
    """Count experiments in storage"""
    return len(get_storage().fetch_benchmark({}))


class TestCreateBenchmark:
    """Test Benchmark creation"""

    @pytest.mark.usefixtures("setup_pickleddb_database")
    def test_create_benchmark_no_storage(self, benchmark_config_py):
        """Test creation if storage is not configured"""
        name = "oopsie_forgot_a_storage"
        host = orion.core.config.storage.database.host

        with OrionState(storage=orion.core.config.storage.to_dict()) as cfg:
            # Reset the Storage and drop instances so that get_storage() would fail.
            cfg.cleanup()
            cfg.singletons = update_singletons()

            # Make sure storage must be instantiated during `get_or_create_benchmark()`
            with pytest.raises(SingletonNotInstantiatedError):
                get_storage()

            get_or_create_benchmark(**benchmark_config_py)

            storage = get_storage()

            assert isinstance(storage, Legacy)
            assert isinstance(storage._db, PickledDB)
            assert storage._db.host == host

    def test_create_benchmark_with_storage(self, benchmark_config_py):
        """Test benchmark instance has the storage configurations"""

        config = copy.deepcopy(benchmark_config_py)
        storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
        with OrionState(storage=storage):
            config["storage"] = storage
            bm = get_or_create_benchmark(**config)

            assert bm.storage_config == config["storage"]

    def test_create_benchmark_bad_storage(self, benchmark_config_py):
        """Test error message if storage is not configured properly"""
        name = "oopsie_bad_storage"
        # Make sure there is no existing storage singleton
        update_singletons()

        with pytest.raises(NotImplementedError) as exc:
            benchmark_config_py["storage"] = {
                "type": "legacy",
                "database": {"type": "idontexist"},
            }
            get_or_create_benchmark(**benchmark_config_py)

        assert (
            "Could not find implementation of AbstractDB, type = 'idontexist'"
            in str(exc.value)
        )

    def test_create_experiment_debug_mode(self, tmp_path, benchmark_config_py):
        """Test that EphemeralDB is used in debug mode whatever the storage config given"""
        update_singletons()

        conf_file = str(tmp_path / "db.pkl")

        config = copy.deepcopy(benchmark_config_py)
        config["storage"] = {
            "type": "legacy",
            "database": {"type": "pickleddb", "host": conf_file},
        }

        get_or_create_benchmark(**config)

        storage = get_storage()

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)

        update_singletons()
        config["storage"] = {"type": "legacy", "database": {"type": "pickleddb"}}
        config["debug"] = True
        get_or_create_benchmark(**config)

        storage = get_storage()

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, EphemeralDB)

    def test_create_benchmark(self, benchmark_config, benchmark_config_py):
        """Test creation with valid configuration"""
        with OrionState():
            bm1 = get_or_create_benchmark(**benchmark_config_py)

            bm2 = get_or_create_benchmark("bm00001")

            assert bm1.configuration == benchmark_config

            assert bm1.configuration == bm2.configuration

    def test_create_with_only_name(self):
        """Test creation with a non-existing benchmark name"""
        with OrionState():
            name = "bm00001"
            with pytest.raises(NoConfigurationError) as exc:
                get_or_create_benchmark(name)

            assert "Benchmark {} does not exist in DB".format(name) in str(exc.value)

    def test_create_with_different_configure(self, benchmark_config_py, caplog):
        """Test creation with same name but different configure"""
        with OrionState():
            config = copy.deepcopy(benchmark_config_py)
            bm1 = get_or_create_benchmark(**config)

            config = copy.deepcopy(benchmark_config_py)
            config["targets"][0]["assess"] = [AverageResult(2)]

            with caplog.at_level(
                logging.WARNING, logger="orion.benchmark.benchmark_client"
            ):
                bm2 = get_or_create_benchmark(**config)

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
                bm3 = get_or_create_benchmark(**config)

            assert bm3.configuration == bm1.configuration
            assert (
                "Benchmark with same name is found but has different configuration, "
                "which will be used for this creation." in caplog.text
            )

    def test_create_with_invalid_algorithms(self, benchmark_config_py):
        """Test creation with a not existed algorithm"""
        with OrionState():

            with pytest.raises(NotImplementedError) as exc:
                benchmark_config_py["algorithms"] = [
                    {"algorithm": {"fake_algorithm": {"seed": 1}}}
                ]
                get_or_create_benchmark(**benchmark_config_py)
            assert "Could not find implementation of BaseAlgorithm" in str(exc.value)

    def test_create_with_deterministic_algorithm(self, benchmark_config_py):
        algorithms = [
            {"algorithm": {"random": {"seed": 1}}},
            {"algorithm": {"gridsearch": {"n_values": 50}}, "deterministic": True},
        ]
        with OrionState():
            config = copy.deepcopy(benchmark_config_py)
            config["algorithms"] = algorithms
            bm = get_or_create_benchmark(**config)

            for study in bm.studies:
                for status in study.status():
                    algo = status["algorithm"]
                    if algo == "gridsearch":
                        assert status["experiments"] == 1
                    else:
                        assert status["experiments"] == study.assessment.task_num

    def test_create_with_invalid_targets(self, benchmark_config_py):
        """Test creation with invalid Task and Assessment"""
        with OrionState():

            with pytest.raises(AttributeError) as exc:
                config = copy.deepcopy(benchmark_config_py)
                config["targets"] = [
                    {"assess": [AverageResult(2)], "task": [DummyTask]}
                ]
                get_or_create_benchmark(**config)

            assert "type object '{}' has no attribute ".format("DummyTask") in str(
                exc.value
            )

            with pytest.raises(AttributeError) as exc:
                config = copy.deepcopy(benchmark_config_py)
                config["targets"] = [
                    {"assess": [DummyAssess], "task": [RosenBrock(25, dim=3)]}
                ]
                get_or_create_benchmark(**config)

            assert "type object '{}' has no attribute ".format("DummyAssess") in str(
                exc.value
            )

    def test_create_with_not_loaded_targets(self, benchmark_config):
        """Test creation with assessment or task does not exist or not loaded"""

        cfg_invalid_assess = copy.deepcopy(benchmark_config)
        cfg_invalid_assess["targets"][0]["assess"]["idontexist"] = {"task_num": 2}

        with OrionState(benchmarks=cfg_invalid_assess):
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(benchmark_config["name"])
            assert "Could not find implementation of BaseAssess" in str(exc.value)

        cfg_invalid_task = copy.deepcopy(benchmark_config)
        cfg_invalid_task["targets"][0]["task"]["idontexist"] = {"max_trials": 2}

        with OrionState(benchmarks=cfg_invalid_task):
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(benchmark_config["name"])
            assert "Could not find implementation of BaseTask" in str(exc.value)

    def test_create_with_not_exist_targets_parameters(self, benchmark_config):
        """Test creation with not existing assessment parameters"""

        benchmark_config["targets"][0]["assess"]["AverageResult"] = {
            "task_num": 2,
            "idontexist": 100,
        }

        with OrionState(benchmarks=benchmark_config):
            with pytest.raises(TypeError) as exc:
                get_or_create_benchmark(benchmark_config["name"])
            assert "__init__() got an unexpected keyword argument 'idontexist'" in str(
                exc.value
            )

    def test_create_from_db_config(self, benchmark_config):
        """Test creation from existing db configubenchmark_configre"""
        with OrionState(benchmarks=copy.deepcopy(benchmark_config)):
            bm = get_or_create_benchmark(benchmark_config["name"])
            assert bm.configuration == benchmark_config

    def test_create_race_condition(
        self, benchmark_config, benchmark_config_py, monkeypatch, caplog
    ):
        """Test creation in race condition"""
        with OrionState(benchmarks=benchmark_config):

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
                bm = benchmark_client.get_or_create_benchmark(**benchmark_config_py)

            assert (
                "Benchmark registration failed. This is likely due to a race condition. "
                "Now rolling back and re-attempting building it." in caplog.text
            )
            assert insert_race_condition.count == 2

            del benchmark_config["_id"]

            assert bm.configuration == benchmark_config
            assert count_benchmarks() == 1
