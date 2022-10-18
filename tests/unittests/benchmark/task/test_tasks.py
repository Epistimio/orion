#!/usr/bin/env python
"""Tests for :mod:`orion.benchmark.task`."""
import inspect

import pytest

from orion.algo.space import Space
from orion.benchmark.task import Branin, CarromTable, EggHolder, HPOBench, RosenBrock
from orion.benchmark.task.hpobench import import_optional


class TestBranin:
    """Test benchmark task branin"""

    def test_creation(self):
        """Test creation"""
        task = Branin(2)
        assert task.max_trials == 2
        assert task.configuration == {"Branin": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = Branin(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = Branin(2)

        assert task.get_search_space() == {"x": "uniform(0, 1, shape=2, precision=10)"}


class TestCarromTable:
    """Test benchmark task CarromTable"""

    def test_creation(self):
        """Test creation"""
        task = CarromTable(2)
        assert task.max_trials == 2
        assert task.configuration == {"CarromTable": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = CarromTable(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = CarromTable(2)

        assert task.get_search_space() == {"x": "uniform(-10, 10, shape=2)"}


class TestEggHolder:
    """Test benchmark task EggHolder"""

    def test_creation(self):
        """Test creation"""
        task = EggHolder(max_trials=2, dim=3)
        assert task.max_trials == 2
        assert task.configuration == {"EggHolder": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = EggHolder(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = EggHolder(2)

        assert task.get_search_space() == {"x": "uniform(-512, 512, shape=2)"}


class TestRosenBrock:
    """Test benchmark task RosenBrock"""

    def test_creation(self):
        """Test creation"""
        task = RosenBrock(max_trials=2, dim=3)
        assert task.max_trials == 2
        assert task.configuration == {"RosenBrock": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = RosenBrock(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = RosenBrock(2)
        assert task.get_search_space() == {"x": "uniform(-5, 10, shape=2)"}


@pytest.mark.skipif(
    import_optional.failed,
    reason="Running without HPOBench",
)
class TestHPOBench:
    """Test benchmark task HPOBenchWrapper"""

    def test_create_with_non_container_benchmark(self):
        """Test to create HPOBench local benchmark"""
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class="hpobench.benchmarks.ml.tabular_benchmark.TabularBenchmark",
            benchmark_kwargs=dict(model="xgb", task_id=168912),
        )
        assert task.max_trials == 2
        assert inspect.isclass(task.hpo_benchmark_cls)
        assert task.configuration == {
            "HPOBench": {
                "hpo_benchmark_class": "hpobench.benchmarks.ml.tabular_benchmark.TabularBenchmark",
                "benchmark_kwargs": {"model": "xgb", "task_id": 168912},
                "objective_function_kwargs": None,
                "max_trials": 2,
            }
        }

    def test_create_with_container_benchmark(self):
        """Test to create HPOBench container benchmark"""
        with pytest.raises(AttributeError) as ex:
            HPOBench(
                max_trials=2,
                hpo_benchmark_class="hpobench.container.benchmark.ml.tabular_benchmarks.TabularBenchmark",
            )
        assert "Can not run containerized benchmark without Singularity" in str(
            ex.value
        )

    def test_call(self):
        """Test to run a local HPOBench benchmark"""
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class="hpobench.benchmarks.ml.tabular_benchmark.TabularBenchmark",
            benchmark_kwargs=dict(model="xgb", task_id=168912),
        )
        params = {
            "colsample_bytree": 1.0,
            "eta": 0.045929204672575,
            "max_depth": 1.0,
            "reg_lambda": 10.079368591308594,
        }

        objectives = task(**params)
        assert objectives == [
            {
                "name": "TabularBenchmark",
                "type": "objective",
                "value": 0.056373193166885674,
            }
        ]

    def test_search_space(self):
        """Test to get task search space"""
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class="hpobench.benchmarks.ml.tabular_benchmark.TabularBenchmark",
            benchmark_kwargs=dict(model="xgb", task_id=168912),
        )
        space = task.get_search_space()

        assert isinstance(space, Space)
        assert len(space) == 4
