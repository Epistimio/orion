import inspect

import pytest

from orion.algo.space import Space
from orion.benchmark.task import HPOBench
from orion.benchmark.task.hpobench import import_optional

hpobench_benchmarks = list()
hpobench_benchmarks.append(
    {
        "type": "tabular",
        "class": "hpobench.container.benchmarks.ml.tabular_benchmark.TabularBenchmark",
        "init_args": dict(model="xgb", task_id=168912),
        "objective_args": dict(),
        "hyperparams": {
            "colsample_bytree": 1.0,
            "eta": 0.045929204672575,
            "max_depth": 1,
            "reg_lambda": 10.079368591308594,
        },
    }
)

hpobench_benchmarks.append(
    {
        "type": "raw",
        "class": "hpobench.container.benchmarks.ml.xgboost_benchmark.XGBoostBenchmark",
        "init_args": dict(task_id=168912),
        "objective_args": dict(),
        "hyperparams": {
            "colsample_bytree": 1.0,
            "eta": 0.045929204672575,
            "max_depth": 1,
            "reg_lambda": 10.079368591308594,
        },
    }
)

"""
# need fix of https://github.com/Epistimio/orion/issues/1018
hpobench_benchmarks.append({
    "type": "surrogate",
    "class": "hpobench.container.benchmarks.surrogates.paramnet_benchmark.ParamNetAdultOnStepsBenchmark",
    "init_args": dict(),
    "objective_args": dict(),
    "hyperparams": {
        "average_units_per_layer_log2": 6.0,
        "batch_size_log2": 5.5,
        "dropout_0": 0.25,
        "dropout_1": 0.25,
        "final_lr_fraction_log2": 1.0
    }
})
"""


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
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class="hpobench.container.benchmarks.ml.tabular_benchmark.TabularBenchmark",
            benchmark_kwargs=dict(model="xgb", task_id=168912),
        )
        assert task.max_trials == 2
        assert inspect.isclass(task.hpo_benchmark_cls)
        assert task.configuration == {
            "HPOBench": {
                "hpo_benchmark_class": "hpobench.container.benchmarks.ml.tabular_benchmark.TabularBenchmark",
                "benchmark_kwargs": {"model": "xgb", "task_id": 168912},
                "objective_function_kwargs": None,
                "max_trials": 2,
            }
        }

    def test_run_locally(self):
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

    @pytest.mark.parametrize("benchmark", hpobench_benchmarks)
    def test_run_singulariys(self, benchmark):
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class=benchmark.get("class"),
            benchmark_kwargs=benchmark.get("init_args"),
            objective_function_kwargs=benchmark.get("objective_args"),
        )

        params = benchmark.get("hyperparams")
        objectives = task(**params)

        assert len(objectives) > 0

    def test_run_singulariy(self):
        task = HPOBench(
            max_trials=2,
            hpo_benchmark_class="hpobench.container.benchmarks.ml.tabular_benchmark.TabularBenchmark",
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
