#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for algos included with orion."""
import copy
import random

import numpy
import pytest

from orion.client import create_experiment, workon
from orion.testing.state import OrionState

storage = {"type": "legacy", "database": {"type": "ephemeraldb"}}


space = {"x": "uniform(-50, 50)"}


space_with_fidelity = {"x": space["x"], "noise": "fidelity(1,10,4)"}


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
}

no_fidelity_algorithms = ["random", "tpe", "gridsearch"]
no_fidelity_algorithm_configs = {
    key: algorithm_configs[key] for key in no_fidelity_algorithms
}

fidelity_only_algorithms = ["asha", "hyperband", "evolutiones"]
fidelity_only_algorithm_configs = {
    key: algorithm_configs[key] for key in fidelity_only_algorithms
}


def rosenbrock(x, noise=None):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789
    if noise:
        noise = (1 - noise / 10) + 0.0001
        z *= random.gauss(0, noise)

    return [
        {"name": "objective", "type": "objective", "value": 4 * z ** 2 + 23.4},
        {"name": "gradient", "type": "gradient", "value": [8 * z]},
    ]


def multidim_rosenbrock(x, noise=None, shape=(2, 1)):
    x = numpy.array(x)
    assert x.shape == shape
    return rosenbrock(x.reshape(-1)[0], noise)


@pytest.mark.parametrize(
    "algorithm",
    fidelity_only_algorithm_configs.values(),
    ids=list(fidelity_only_algorithm_configs.keys()),
)
def test_missing_fidelity(algorithm):
    """Test a simple usage scenario."""
    with pytest.raises(RuntimeError) as exc:
        workon(rosenbrock, space, algorithms=algorithm, max_trials=100)

    assert "https://orion.readthedocs.io/en/develop/user/algorithms.html" in str(
        exc.value
    )


@pytest.mark.parametrize(
    "algorithm",
    no_fidelity_algorithm_configs.values(),
    ids=list(no_fidelity_algorithm_configs.keys()),
)
def test_simple(algorithm):
    """Test a simple usage scenario."""
    max_trials = 30
    exp = workon(rosenbrock, space, algorithms=algorithm, max_trials=max_trials)

    assert exp.max_trials == max_trials
    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) == max_trials
    assert trials[-1].status == "completed"

    best_trial = sorted(trials, key=lambda trial: trial.objective.value)[0]
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
def test_cardinality_stop(algorithm):
    """Test when algo needs to stop because all space is explored (dicrete space)."""
    discrete_space = copy.deepcopy(space)
    discrete_space["x"] = "uniform(-10, 5, discrete=True)"
    exp = workon(rosenbrock, discrete_space, algorithms=algorithm, max_trials=30)

    trials = exp.fetch_trials()
    assert len(trials) == 16
    assert trials[-1].status == "completed"

    discrete_space["x"] = "loguniform(0.1, 1, precision=1)"
    exp = workon(rosenbrock, discrete_space, algorithms=algorithm, max_trials=30)
    print(exp.space.cardinality)

    trials = exp.fetch_trials()
    assert len(trials) == 10
    assert trials[-1].status == "completed"


@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_with_fidelity(algorithm):
    """Test a scenario with fidelity."""
    exp = workon(rosenbrock, space_with_fidelity, algorithms=algorithm, max_trials=30)

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) >= 30
    assert trials[29].status == "completed"

    trials = [trial for trial in trials if trial.status == "completed"]
    results = [trial.objective.value for trial in trials]
    best_trial = next(iter(sorted(trials, key=lambda trial: trial.objective.value)))

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 10
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value in [1, 2, 5, 10]
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
    assert len(trials) >= 25  # Grid-search is a bit smaller
    completed_trials = exp.fetch_trials_by_status("completed")
    assert len(completed_trials) >= MAX_TRIALS or len(completed_trials) == len(trials)

    trials = completed_trials
    results = [trial.objective.value for trial in trials]
    best_trial = next(iter(sorted(trials, key=lambda trial: trial.objective.value)))

    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 10
    assert len(best_trial.params) == 2
    fidelity = best_trial._params[0]
    assert fidelity.name == "noise"
    assert fidelity.type == "fidelity"
    assert fidelity.value in [1, 2, 5, 10]
    param = best_trial._params[1]
    assert param.name == "x"
    assert param.type == "real"


@pytest.mark.skip("Enable back when EVC is supported again")
@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_with_evc(algorithm):
    """Test a scenario where algos are warm-started with EVC."""

    with OrionState(storage={"type": "legacy", "database": {"type": "PickledDB"}}):
        base_exp = create_experiment(
            name="exp",
            space=space_with_fidelity,
            algorithms=algorithm_configs["random"],
            max_trials=10,
        )
        base_exp.workon(rosenbrock, max_trials=10)

        exp = create_experiment(
            name="exp",
            space=space_with_fidelity,
            algorithms=algorithm,
            max_trials=30,
            branching={"branch_from": "exp", "enable": True},
        )

        assert exp.version == 2

        exp.workon(rosenbrock, max_trials=30)

        assert exp.configuration["algorithms"] == algorithm

        trials = exp.fetch_trials(with_evc_tree=False)

        # Some algo may not be able to suggest exactly 30 trials (ex: hyperband)
        assert len(trials) >= 20

        trials_with_evc = exp.fetch_trials(with_evc_tree=True)
        assert len(trials_with_evc) >= 30
        assert len(trials_with_evc) - len(trials) == 10

        completed_trials = [
            trial for trial in trials_with_evc if trial.status == "completed"
        ]
        assert len(completed_trials) == 30

        results = [trial.objective.value for trial in completed_trials]
        best_trial = next(
            iter(sorted(completed_trials, key=lambda trial: trial.objective.value))
        )

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
def test_parallel_workers(algorithm):
    """Test parallel execution with joblib"""
    MAX_TRIALS = 30
    ASHA_UGLY_FIX = 10
    with OrionState() as cfg:  # Using PickledDB

        name = "{}_exp".format(list(algorithm.keys())[0])

        exp = create_experiment(
            name=name,
            space=space_with_fidelity,
            algorithms=algorithm,
        )

        exp.workon(rosenbrock, max_trials=MAX_TRIALS, n_workers=2)

        assert exp.configuration["algorithms"] == algorithm

        trials = exp.fetch_trials()
        assert len(trials) >= MAX_TRIALS

        completed_trials = [trial for trial in trials if trial.status == "completed"]
        assert MAX_TRIALS <= len(completed_trials) <= MAX_TRIALS + 2

        results = [trial.objective.value for trial in completed_trials]
        best_trial = next(
            iter(sorted(completed_trials, key=lambda trial: trial.objective.value))
        )

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
