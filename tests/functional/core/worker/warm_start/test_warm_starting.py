""" Functional tests for the warm-starting feature. """
from __future__ import annotations

import functools
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from typing_extensions import ParamSpec

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.algo.tpe import TPE
from orion.client import build_experiment, workon
from orion.client.experiment import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.knowledge_base import KnowledgeBase
from orion.storage.base import BaseStorageProtocol, setup_storage

# Function to create a space.
_space: Callable[[dict], Space] = SpaceBuilder().build


def simple_quadratic(
    a: float = 1, b: float = -2, c: float = 1
) -> Callable[[float], float]:
    """A simple quadratic function."""
    return lambda x: a * x**2 + b * x + c


def _temp_storage(pickle_path: Path | str) -> BaseStorageProtocol:
    """Creates a temporary storage."""
    return setup_storage(
        {
            "type": "legacy",
            "database": {"type": "pickleddb", "host": str(pickle_path)},
        }
    )


@pytest.mark.parametrize("algo", [TPE])
def test_warm_starting_helps(
    algo: type[BaseAlgorithm],
    storage: BaseStorageProtocol,
):
    """Integration test. Shows that warm-starting helps in a simple task."""
    algo_config = {"of_type": algo.__qualname__.lower(), "seed": 42}

    # TODO: There still seems to be some randomness in this test, despite the algorithms being
    # seeded..
    np.random.seed(123)
    random.seed(123)

    source_task = simple_quadratic(a=1, b=-2, c=1)
    target_task = simple_quadratic(a=1, b=-2, c=1)
    # Note: minimum is at x = 1, with y = 0
    # Here we could prime the source task with points that should be super useful, since they
    # are really close to the minimum:
    source_space: Space = _space({"x": "uniform(0, 10)"})
    target_space: Space = _space({"x": "uniform(0, 10)"})

    n_source_trials = 50  # Number of trials from the source task
    max_trials = 10  # Number of trials in the target task

    source_experiment = build_experiment(
        name="source_exp",
        space=source_space,
        algorithm={"of_type": "random", "seed": 42},
        storage=storage,
        max_trials=n_source_trials,
    )
    source_experiment.workon(_wrap(source_task))
    # Create the Knowledge base by passing the now-filled Storage.
    knowledge_base = KnowledgeBase(storage=storage)
    assert knowledge_base.n_stored_experiments == 1

    without_warm_starting = workon(
        _wrap(target_task),
        name="witout_warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithm=algo_config,
        knowledge_base=None,
    )
    assert len(without_warm_starting.fetch_trials()) == max_trials

    with_warm_starting = workon(
        _wrap(target_task),
        name="with_warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithm=algo_config,
        knowledge_base=knowledge_base,
    )
    assert len(with_warm_starting.fetch_trials()) == max_trials

    best_trial_with = _get_best_trial(with_warm_starting)
    best_trial_without = _get_best_trial(without_warm_starting)

    objective_with = _get_objective(best_trial_with)
    objective_without = _get_objective(best_trial_without)
    assert objective_with < objective_without


@pytest.mark.parametrize("algo", [TPE])
def test_warm_start_benchmarking(algo: type[BaseAlgorithm], tmp_path: Path):
    """Integration test. Compares the performance in the three cases (cold, warm, hot)-starting.

    - Cold-start: Optimize the target task with no prior knowledge (lower bound);
    - Warm-start: Optimize the target task with some prior knowledge from "related" tasks;
    - Hot-start:  Optimize the target task with some prior knowledge from the *same* task (upper
                  bound).
    """
    algo_config = {"of_type": algo.__qualname__.lower(), "seed": 42}

    source_task = simple_quadratic(a=1, b=-2, c=1)
    target_task = simple_quadratic(a=1, b=-2, c=1)

    source_space: Space = _space({"x": "uniform(-5, 5)"})
    target_space: Space = _space({"x": "uniform(-5, 5)"})

    n_source_trials = 20  # Number of trials from the source task
    max_trials = 20  # Number of trials in the target task

    warm_start_storage = _temp_storage(tmp_path / "warm.pkl")
    # Populate the warm-start storage with some data from the source task.
    source_experiment = build_experiment(
        name="source",
        space=source_space,
        storage=warm_start_storage,
        max_trials=n_source_trials,
        algorithm={"of_type": "random", "seed": 42},
    )
    source_experiment.workon(_wrap(source_task))
    warm_start_kb = KnowledgeBase(warm_start_storage)
    assert warm_start_kb.n_stored_experiments == 1

    hot_start_storage = _temp_storage(tmp_path / "hot.pkl")
    # Populate the hot-start storage with some data from the *target* task.
    target_prior_experiment = build_experiment(
        name="target_prior",
        space=target_space,
        storage=hot_start_storage,
        max_trials=n_source_trials,
        algorithm={"of_type": "random", "seed": 42},
    )
    target_prior_experiment.workon(_wrap(target_task))
    hot_start_kb = KnowledgeBase(hot_start_storage)

    # Execute the three different experiments:
    # Cold start: No prior information.
    cold_start = workon(
        _wrap(target_task),
        name="cold_start",
        space=target_space,
        max_trials=max_trials,
        algorithm=algo_config,
        knowledge_base=None,
    )
    # Warm-start: Prior information from the source task.
    warm_start = workon(
        _wrap(target_task),
        name="warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithm=algo_config,
        knowledge_base=warm_start_kb,
    )
    # Hot-start: Prior information from the target task.
    hot_start = workon(
        _wrap(target_task),
        name="hot_start",
        space=target_space,
        max_trials=max_trials,
        algorithm=algo_config,
        knowledge_base=hot_start_kb,
    )

    cold_objective = _get_best_trial_objective(cold_start)
    warm_objective = _get_best_trial_objective(warm_start)
    hot_objective = _get_best_trial_objective(hot_start)
    assert hot_objective <= warm_objective
    assert warm_objective <= cold_objective


P = ParamSpec("P")


def _wrap(objective_fn: Callable[P, float]) -> Callable[P, list[dict]]:
    """Adds some common boilerplate to this objective function."""
    return _with_results(objective_fn)


def _with_results(objective_fn: Callable[P, float]) -> Callable[P, list[dict]]:
    """Adds the boilerplate to create a list of 'Results' dictionary from the objective."""

    @functools.wraps(objective_fn)
    def _with_results(*args: P.args, **kwargs: P.kwargs) -> list[dict]:
        objective = objective_fn(*args, **kwargs)
        return [dict(name="objective", type="objective", value=objective)]

    return _with_results


def _get_objective(trial: Trial) -> float:
    objective: Trial.Result | None = trial.objective
    if objective is not None:
        return objective.value
    return float("inf")


def _get_best_trial(experiment: ExperimentClient) -> Trial:
    trials: list[Trial] = experiment.fetch_trials_by_status(status="completed")
    return min(trials, key=_get_objective)


def _get_best_trial_objective(experiment: ExperimentClient) -> float:
    best_trial = _get_best_trial(experiment)
    assert best_trial.objective is not None
    return best_trial.objective.value
