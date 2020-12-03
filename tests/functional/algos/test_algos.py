#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for algos included with orion."""
import copy
import random

import pytest

from orion.client import workon

storage = {"type": "legacy", "database": {"type": "ephemeraldb"}}


space = {"x": "uniform(-50, 50)"}


space_with_fidelity = {"x": space["x"], "noise": "fidelity(1,10,4)"}


algorithm_configs = {
    "random": {"random": {"seed": 1}},
    "tpe": {
        "tpe": {
            "seed": 1,
            "n_initial_points": 20,
            "n_ei_candidates": 24,
            "gamma": 0.25,
            "equal_weight": False,
            "prior_weight": 1.0,
            "full_weight_num": 25,
        }
    },
    "asha": {
        "asha": {
            "seed": 1,
            "num_rungs": 4,
            "num_brackets": 1,
            "grace_period": None,
            "max_resources": None,
            "reduction_factor": None,
        }
    },
    "hyperband": {"hyperband": {"repetitions": 5, "seed": 1}},
}

no_fidelity_algorithms = ["random", "tpe"]
no_fidelity_algorithm_configs = {
    key: algorithm_configs[key] for key in no_fidelity_algorithms
}

fidelity_only_algorithms = ["asha", "hyperband"]
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
    max_trials = 100
    exp = workon(rosenbrock, space, algorithms=algorithm, max_trials=max_trials)

    assert exp.max_trials == max_trials
    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) == max_trials
    assert trials[-1].status == "completed"

    best_trial = sorted(trials, key=lambda trial: trial.objective.value)[0]
    assert best_trial.objective.name == "objective"
    assert abs(best_trial.objective.value - 23.4) < 0.1
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
    exp = workon(rosenbrock, discrete_space, algorithms=algorithm, max_trials=100)

    trials = exp.fetch_trials()
    assert len(trials) == 16
    assert trials[-1].status == "completed"


@pytest.mark.parametrize(
    "algorithm", algorithm_configs.values(), ids=list(algorithm_configs.keys())
)
def test_with_fidelity(algorithm):
    """Test a scenario with fidelity."""
    exp = workon(rosenbrock, space_with_fidelity, algorithms=algorithm, max_trials=100)

    assert exp.configuration["algorithms"] == algorithm

    trials = exp.fetch_trials()
    assert len(trials) <= 100
    assert trials[-1].status == "completed"

    results = [trial.objective.value for trial in trials]
    best_trial = next(iter(sorted(trials, key=lambda trial: trial.objective.value)))

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
