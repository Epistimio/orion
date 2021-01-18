#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.benchmark_client`."""

import pytest

import orion.core
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import CarromTable, RosenBrock
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.utils.singleton import SingletonNotInstantiatedError, update_singletons
from orion.storage.base import get_storage
from orion.storage.legacy import Legacy
from orion.testing import OrionState


class DummyTask:
    """Dummy invalid benchmark task"""

    pass


class DummyAssess:
    """Dummy invalid benchmark assessment"""

    pass


class TestCreateBenchmark:
    """Test Benchmark creation"""

    @pytest.mark.usefixtures("setup_pickleddb_database")
    def test_create_experiment_no_storage(self, benchmark_algorithms):
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

            bm1 = get_or_create_benchmark(
                "bm00001",
                algorithms=benchmark_algorithms,
                targets=[
                    {
                        "assess": [AverageResult(2), AverageRank(2)],
                        "task": [RosenBrock(25, dim=3), CarromTable(20)],
                    }
                ],
            )

            storage = get_storage()

            assert isinstance(storage, Legacy)
            assert isinstance(storage._db, PickledDB)
            assert storage._db.host == host

    def test_create_experiment_bad_storage(self, benchmark_algorithms):
        """Test error message if storage is not configured properly"""
        name = "oopsie_bad_storage"
        # Make sure there is no existing storage singleton
        update_singletons()

        with pytest.raises(NotImplementedError) as exc:
            get_or_create_benchmark(
                "bm00001",
                algorithms=benchmark_algorithms,
                targets=[
                    {
                        "assess": [AverageResult(2), AverageRank(2)],
                        "task": [RosenBrock(25, dim=3), CarromTable(20)],
                    }
                ],
                storage={"type": "legacy", "database": {"type": "idontexist"}},
            )

        assert (
            "Could not find implementation of AbstractDB, type = 'idontexist'"
            in str(exc.value)
        )

    def test_create_experiment_debug_mode(self, tmp_path, benchmark_algorithms):
        """Test that EphemeralDB is used in debug mode whatever the storage config given"""
        update_singletons()

        conf_file = str(tmp_path / "db.pkl")

        get_or_create_benchmark(
            "bm00001",
            algorithms=benchmark_algorithms,
            targets=[
                {
                    "assess": [AverageResult(2), AverageRank(2)],
                    "task": [RosenBrock(25, dim=3), CarromTable(20)],
                }
            ],
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": conf_file},
            },
        )

        storage = get_storage()

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, PickledDB)

        update_singletons()

        get_or_create_benchmark(
            "bm00001",
            algorithms=benchmark_algorithms,
            targets=[
                {
                    "assess": [AverageResult(2), AverageRank(2)],
                    "task": [RosenBrock(25, dim=3), CarromTable(20)],
                }
            ],
            storage={"type": "legacy", "database": {"type": "pickleddb"}},
            debug=True,
        )

        storage = get_storage()

        assert isinstance(storage, Legacy)
        assert isinstance(storage._db, EphemeralDB)

    def test_create_benchmark(self, benchmark_algorithms):
        """Test creation with valid configuration"""
        with OrionState():
            bm1 = get_or_create_benchmark(
                "bm00001",
                algorithms=benchmark_algorithms,
                targets=[
                    {
                        "assess": [AverageResult(2), AverageRank(2)],
                        "task": [RosenBrock(25, dim=3), CarromTable(20)],
                    }
                ],
            )

            bm2 = get_or_create_benchmark("bm00001")

            cfg = {
                "name": "bm00001",
                "algorithms": benchmark_algorithms,
                "targets": [
                    {
                        "assess": [
                            {
                                "orion-benchmark-assessment-averageresult-AverageResult": {
                                    "task_num": 2
                                }
                            },
                            {
                                "orion-benchmark-assessment-averagerank-AverageRank": {
                                    "task_num": 2
                                }
                            },
                        ],
                        "task": [
                            {
                                "orion-benchmark-task-rosenbrock-RosenBrock": {
                                    "dim": 3,
                                    "max_trials": 25,
                                }
                            },
                            {
                                "orion-benchmark-task-carromtable-CarromTable": {
                                    "max_trials": 20
                                }
                            },
                        ],
                    }
                ],
            }
            assert bm1.configuration == cfg

            assert bm1.configuration == bm2.configuration

    def test_create_with_only_name(self):
        """Test creation with a non-existing benchmark name"""
        with OrionState():
            name = "bm00001"
            with pytest.raises(NoConfigurationError) as exc:
                get_or_create_benchmark(name)

            assert "Benchmark {} does not exist in DB".format(name) in str(exc.value)

    def test_create_with_invalid_algorithms(self):
        """Test creation with a not existed algorithm"""
        with OrionState():
            name = "bm00001"
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(
                    name,
                    algorithms=[{"fake_algorithm": {"seed": 1}}],
                    targets=[
                        {
                            "assess": [AverageResult(2), AverageRank(2)],
                            "task": [RosenBrock(25, dim=3), CarromTable(20)],
                        }
                    ],
                )
            assert "Could not find implementation of BaseAlgorithm" in str(exc.value)

    def test_create_with_invalid_targets(self, benchmark_algorithms):
        """Test creation with invalid Task and Assessment"""
        with OrionState():
            name = "bm00001"
            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(
                    name,
                    algorithms=benchmark_algorithms,
                    targets=[{"assess": [AverageResult(2)], "task": [DummyTask]}],
                )
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format("DummyTask") in str(
                exc.value
            )

            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(
                    name,
                    algorithms=benchmark_algorithms,
                    targets=[
                        {"assess": [DummyAssess], "task": [RosenBrock(25, dim=3)]}
                    ],
                )
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format("DummyAssess") in str(
                exc.value
            )
