#!/usr/bin/env python
"""Perform a functional test for algos included with orion."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy
import pytest

from orion.algo.pbt.pb2_utils import import_optional as pb2_import_optional
from orion.client import create_experiment, workon
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.module_import import ImportOptional
from orion.core.worker.primary_algo import SpaceTransformAlgoWrapper
from orion.core.worker.trial import Trial
from orion.storage.base import BaseStorageProtocol

storage = {"type": "legacy", "database": {"type": "ephemeraldb"}}


algorithm_configs = {
    "random": {"random": {"seed": 1}},
    "gridsearch": {"gridsearch": {"n_values": 40}},
    "tpe": {
        "tpe": {
            "seed": 1,
            "n_initial_points": 20,
            "n_ei_candidates": 24,
            "gamma": 0.25,
            "equal_weight": False,
            "prior_weight": 1.0,
            "full_weight_num": 25,
            "max_retry": 100,
            "parallel_strategy": {
                "of_type": "StatusBasedParallelStrategy",
                "strategy_configs": {
                    "broken": {
                        "of_type": "MaxParallelStrategy",
                    },
                },
            },
        }
    },
    "asha": {"asha": {"seed": 1, "num_rungs": 4, "num_brackets": 1, "repetitions": 2}},
    "hyperband": {"hyperband": {"repetitions": 5, "seed": 1}},
    "evolutiones": {
        "evolutiones": {
            "mutate": None,
            "repetitions": 5,
            "nums_population": 20,
            "seed": 1,
            "max_retries": 100,
        }
    },
    "pbt": {
        "pbt": {
            "seed": 1,
            "generations": 5,
            "population_size": 10,
            "exploit": {
                "exploit_configs": [
                    {
                        "candidate_pool_ratio": 0.2,
                        "min_forking_population": 5,
                        "of_type": "BacktrackExploit",
                        "truncation_quantile": 0.9,
                    },
                    {
                        "candidate_pool_ratio": 0.2,
                        "min_forking_population": 5,
                        "of_type": "TruncateExploit",
                        "truncation_quantile": 0.8,
                    },
                ],
                "of_type": "PipelineExploit",
            },
            "explore": {
                "explore_configs": [
                    {"of_type": "ResampleExplore", "probability": 0.2},
                    {"factor": 1.2, "of_type": "PerturbExplore", "volatility": 0.0001},
                ],
                "of_type": "PipelineExplore",
            },
            "fork_timeout": 60,
        }
    },
    "pb2": {
        "pb2": {
            "seed": 1,
            "generations": 2,
            "population_size": 10,
            "exploit": {
                "exploit_configs": [
                    {
                        "candidate_pool_ratio": 0.5,
                        "min_forking_population": 5,
                        "of_type": "BacktrackExploit",
                        "truncation_quantile": 0.9,
                    },
                    {
                        "candidate_pool_ratio": 0.5,
                        "min_forking_population": 5,
                        "of_type": "TruncateExploit",
                        "truncation_quantile": 0.8,
                    },
                ],
                "of_type": "PipelineExploit",
            },
            "fork_timeout": 60,
        }
    },
}


def xfail_if_not_installed(value: dict, import_optional: ImportOptional):
    name = next(iter(value.keys()))
    return pytest.param(
        value,
        marks=pytest.mark.xfail(
            condition=import_optional.failed,
            reason=f"{name} dependency is requered for these tests",
            raises=ImportError,
        ),
    )


algorithm_configs["pb2"] = xfail_if_not_installed(
    algorithm_configs["pb2"], pb2_import_optional
)


no_fidelity_algorithms = ["random", "tpe", "gridsearch"]
no_fidelity_algorithm_configs = {
    key: algorithm_configs[key] for key in no_fidelity_algorithms
}

fidelity_only_algorithms = ["asha", "hyperband", "evolutiones", "pbt", "pb2"]
fidelity_only_algorithm_configs = {
    key: algorithm_configs[key] for key in fidelity_only_algorithms
}

branching_algorithms = ["pbt", "pb2"]
branching_algorithm_configs = {
    key: algorithm_configs[key] for key in branching_algorithms
}


from orion.benchmark.task.base import BenchmarkTask


class CustomRosenbrock(BenchmarkTask):
    def __init__(self, max_trials: int = 30, with_fidelity: bool = False):
        super().__init__(max_trials)
        self.with_fidelity = with_fidelity

    def call(self, x: float, noise: float | None = None) -> list[dict]:
        """Evaluate partial information of a quadratic."""
        z = x - 34.56789
        if noise is not None:
            noise = (1 - noise / 10) + 0.0001
            z *= random.gauss(0, noise)
        y = 4 * z**2 + 23.4
        dy_dx = 8 * z
        return [
            {"name": "objective", "type": "objective", "value": y},
            {"name": "gradient", "type": "gradient", "value": dy_dx},
        ]

    def get_search_space(self) -> dict[str, str]:
        space = {"x": "uniform(-50, 50)"}
        if self.with_fidelity:
            space["noise"] = "fidelity(1,10,4)"
        return space


class MultiDimRosenbrock(CustomRosenbrock):
    def __init__(
        self,
        max_trials: int = 30,
        with_fidelity: bool = False,
        shape: tuple[int, ...] = (2, 1),
    ):
        super().__init__(max_trials, with_fidelity)
        self.shape = shape

    def get_search_space(self) -> dict[str, str]:
        space = super().get_search_space()
        space["x"] = "uniform(-50, 50, shape=(2, 1))"
        return space

    def call(self, x: float | numpy.ndarray, noise: float | None = None) -> list[dict]:
        x = numpy.array(x)
        assert x.shape == self.shape
        x_0: float = x.reshape(-1)[0]
        return super().call(x_0, noise=noise)


rosenbrock = CustomRosenbrock(max_trials=30, with_fidelity=False)
rosenbrock_with_fidelity = CustomRosenbrock(max_trials=30, with_fidelity=True)

space = rosenbrock.get_search_space()
space_with_fidelity = rosenbrock_with_fidelity.get_search_space()

multidim_rosenbrock = MultiDimRosenbrock(max_trials=30, with_fidelity=False)


def branching_rosenbrock(
    x: float, trial: Trial, noise: float | None = None
) -> list[dict]:
    with open(os.path.join(trial.working_dir, "hist.txt"), "a") as f:
        f.write(trial.params_repr() + "\n")

    return rosenbrock(x, noise)


@pytest.mark.parametrize(
    "algorithm",
    fidelity_only_algorithm_configs.values(),
    ids=list(fidelity_only_algorithm_configs.keys()),
)
def test_missing_fidelity(algorithm: dict):
    """Test a simple usage scenario."""
    task = CustomRosenbrock(max_trials=30, with_fidelity=False)
    with pytest.raises(RuntimeError) as exc:
        workon(task, task.get_search_space(), algorithms=algorithm, max_trials=100)

    assert "https://orion.readthedocs.io/en/develop/user/algorithms.html" in str(
        exc.value
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        v
        if k != "gridsearch"
        else pytest.param(
            v, marks=pytest.mark.skip(reason="gridsearch is misbehaving atm.")
        )
        for k, v in no_fidelity_algorithm_configs.items()
    ],
    ids=list(no_fidelity_algorithm_configs.keys()),
)
def test_simple(algorithm: dict):
    """Test a simple usage scenario."""
    max_trials = 30
    exp = workon(rosenbrock, space, algorithms=algorithm, max_trials=max_trials)

    assert exp.max_trials == max_trials
    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) == max_trials
    assert trials[-1].status == "completed"

    assert all(trial.objective is not None for trial in trials)
    best_trial = min(trials, key=lambda trial: trial.objective.value)
    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 15
    assert len(best_trial.params) == 1
    param = best_trial._params[0]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.parametrize(
    "algorithm",
    no_fidelity_algorithm_configs.values(),
    ids=list(no_fidelity_algorithm_configs.keys()),
)
def test_cardinality_stop_uniform(algorithm: dict):
    """Test when algo needs to stop because all space is explored (discrete space)."""
    discrete_space = copy.deepcopy(space)
    discrete_space["x"] = "uniform(-10, 5, discrete=True)"
    exp = workon(rosenbrock, discrete_space, algorithms=algorithm, max_trials=30)

    trials = exp.fetch_trials()
    assert len(trials) == 16
    assert trials[-1].status == "completed"


@pytest.mark.parametrize(
    "algorithm",
    no_fidelity_algorithm_configs.values(),
    ids=list(no_fidelity_algorithm_configs.keys()),
)
def test_cardinality_stop_loguniform(algorithm: dict):
    """Test when algo needs to stop because all space is explored (loguniform space)."""
    discrete_space = SpaceBuilder().build({"x": "loguniform(0.1, 1, precision=1)"})

    max_trials = 30
    exp = workon(
        rosenbrock, space=discrete_space, algorithms=algorithm, max_trials=max_trials
    )
    algo_wrapper: SpaceTransformAlgoWrapper = exp.algorithms
    assert algo_wrapper.space == discrete_space
    assert algo_wrapper.algorithm.is_done
    assert algo_wrapper.is_done

    trials = exp.fetch_trials()
    if algo_wrapper.algorithm.space.cardinality == 10:
        # BUG: See https://github.com/Epistimio/orion/issues/865
        # The algo (e.g. GridSearch) believes it has exhausted the space cardinality and exits early
        # but that's incorrect! The transformed space should have a different cardinality than the
        # original space.
        assert len(trials) <= 10
    else:
        assert len(trials) == 10
    assert trials[-1].status == "completed"


@pytest.mark.parametrize(
    "algorithm",
    algorithm_configs.values(),
    ids=list(algorithm_configs.keys()),
)
def test_with_fidelity(algorithm: dict):
    """Test a scenario with fidelity."""
    exp = workon(
        rosenbrock_with_fidelity,
        space_with_fidelity,
        algorithms=algorithm,
        max_trials=30,
    )

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) >= 30 or exp.algorithms.is_done
    assert trials[-1].status == "completed"

    trials = [trial for trial in trials if trial.status == "completed"]
    assert all(trial.objective is not None for trial in trials)
    best_trial = min(trials, key=lambda trial: trial.objective.value)

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 10
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value in exp.space["noise"]
    param = best_trial._params[1]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_with_multidim(algorithm):
    """Test a scenario with a dimension shape > 1."""
    space = copy.deepcopy(space_with_fidelity)
    space["x"] = "uniform(-50, 50, shape=(2, 1))"
    MAX_TRIALS = 30

    exp = workon(
        multidim_rosenbrock, space, algorithms=algorithm, max_trials=MAX_TRIALS
    )

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) >= 25 or exp.algorithms.is_done
    completed_trials = exp.fetch_trials_by_status("completed")
    assert len(completed_trials) >= MAX_TRIALS or len(completed_trials) == len(trials)

    trials = completed_trials
    results = [trial.objective.value for trial in trials]
    assert all(trial.objective is not None for trial in trials)
    best_trial = min(trials, key=lambda trial: trial.objective.value)

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 10
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value in exp.space["noise"]
    param = best_trial._params[1]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.skip("Enable back when EVC is supported again")
@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_with_evc(algorithm, storage):
    """Test a scenario where algos are warm-started with EVC."""

    base_exp = create_experiment(
        name="exp",
        space=space_with_fidelity,
        algorithms=algorithm_configs["random"],
        max_trials=10,
        storage=storage,
    )
    base_exp.workon(rosenbrock, max_trials=10)

    exp = create_experiment(
        name="exp",
        space=space_with_fidelity,
        algorithms=algorithm,
        max_trials=30,
        storage=storage,
        branching={"branch_from": "exp", "enable": True},
    )

    assert exp.version == 2

    exp.workon(rosenbrock, max_trials=30)

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials(with_evc_tree=False)

    # Some algo may not be able to suggest exactly 30 trials (ex: hyperband)
    assert len(trials) >= 20

    trials_with_evc = exp.fetch_trials(with_evc_tree=True)
    assert len(trials_with_evc) >= 30 or exp.algorithms.is_done
    assert len(trials_with_evc) - len(trials) == 10

    completed_trials = [
        trial for trial in trials_with_evc if trial.status == "completed"
    ]
    assert len(completed_trials) == 30

    results = [trial.objective.value for trial in completed_trials]
    assert all(trial.objective is not None for trial in completed_trials)
    best_trial = min(completed_trials, key=lambda trial: trial.objective.value)

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 1e-5
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value == 10
    param = best_trial._params[1]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_parallel_workers(algorithm, storage):
    """Test parallel execution with joblib"""
    MAX_TRIALS = 30
    ASHA_UGLY_FIX = 10

    name = f"{list(algorithm.keys())[0]}_exp"

    exp = create_experiment(
        name=name, space=space_with_fidelity, algorithms=algorithm, storage=storage
    )

    exp.workon(rosenbrock, max_trials=MAX_TRIALS, n_workers=2)

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) >= MAX_TRIALS or exp.algorithms.is_done

    completed_trials = [trial for trial in trials if trial.status == "completed"]
    assert MAX_TRIALS <= len(completed_trials) <= MAX_TRIALS + 2

    results = [trial.objective.value for trial in completed_trials]
    assert all(trial.objective is not None for trial in completed_trials)
    best_trial = min(completed_trials, key=lambda trial: trial.objective.value)

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 1e-5 + ASHA_UGLY_FIX
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value + ASHA_UGLY_FIX >= 1
    param = best_trial._params[1]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.parametrize(
    "algorithm",
    branching_algorithm_configs.values(),
    ids=list(branching_algorithm_configs.keys()),
)
def test_branching_algos(
    algorithm: dict[str, dict], storage: BaseStorageProtocol, tmp_path: Path
):

    exp = create_experiment(
        name="exp",
        space=space_with_fidelity,
        algorithms=algorithm,
        working_dir=tmp_path,
    )

    exp.workon(branching_rosenbrock, n_workers=2, trial_arg="trial")

    def build_params_hist(trial: Trial) -> list[str]:
        params = [trial.params_repr()]
        while trial.parent:
            trial = exp.algorithms.registry[trial.parent]
            params.append(trial.params_repr())
        return params[::-1]

    for trial in exp.fetch_trials():
        params = build_params_hist(trial)
        # TODO: This assumes algo.fidelities which may be specific to PBT...
        assert (
            len(params)
            == exp.algorithms.algorithm.fidelities.index(trial.params["noise"]) + 1
        )
        with open(os.path.join(trial.working_dir, "hist.txt")) as f:
            assert "\n".join(params) == f.read().strip("\n")
