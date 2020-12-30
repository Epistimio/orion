#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.benchmark_client`."""

import pytest

from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import CarromTable, RosenBrock
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.utils.tests import OrionState


class DummyTask:
    """Dummy invalid benchmark task"""

    pass


class DummyAssess:
    """Dummy invalid benchmark assessment"""

    pass


@pytest.fixture
def algorithms():
    """Return a list of algorithms suitable for Orion experiment"""
    return [{"random": {"seed": 1}}, {"tpe": {"seed": 1}}]


class TestCreateBenchmark:
    """Test Benchmark creation"""

    def test_create_benchmark(self, algorithms):
        """Test creation with valid configuration"""
        with OrionState():
            bm1 = get_or_create_benchmark(
                "bm00001",
                algorithms=algorithms,
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
                "algorithms": algorithms,
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
        """Test creation with a not existed benchmark name"""
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

    def test_create_with_invalid_targets(self, algorithms):
        """Test creation with invalid Task and Assessment"""
        with OrionState():
            name = "bm00001"
            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(
                    name,
                    algorithms=algorithms,
                    targets=[{"assess": [AverageResult(2)], "task": [DummyTask]}],
                )
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format("DummyTask") in str(
                exc.value
            )

            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(
                    name,
                    algorithms=algorithms,
                    targets=[
                        {"assess": [DummyAssess], "task": [RosenBrock(25, dim=3)]}
                    ],
                )
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format("DummyAssess") in str(
                exc.value
            )
