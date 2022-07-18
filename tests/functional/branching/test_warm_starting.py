""" Functional tests for the warm-starting feature. """
from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar

import pytest
from typing_extensions import ParamSpec
from unittests.core.worker.warm_start.test_knowledge_base import (
    DummyKnowledgeBase,
    add_result,
)
from unittests.core.worker.warm_start.test_multi_task import create_dummy_kb

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.algo.tpe import TPE
from orion.client import build_experiment, workon
from orion.client.experiment import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.experiment_config import ExperimentInfo

# Function to create a space.
_space: Callable[[dict], Space] = SpaceBuilder().build

previous_spaces: list[Space] = [
    _space({"x": "uniform(0, 5)"}),
    _space({"x": "uniform(1, 6)"}),
    _space({"x": "uniform(2, 7)"}),
    _space({"x": "uniform(3, 8)"}),
    _space({"x": "uniform(4, 9)"}),
]
target_space = _space({"x": "uniform(0, 10)"})


def simple_quadratic(
    a: float = 1, b: float = -2, c: float = 1
) -> Callable[[float], float]:
    """A simple quadratic function."""
    return lambda x: a * x**2 + b * x + c


@pytest.mark.parametrize("algo", [TPE])
def test_warm_starting_helps(algo: type[BaseAlgorithm]):
    """Integration test. Shows that warm-starting helps in a simple task."""
    source_space: Space = _space({"x": "uniform(0, 10)"})
    source_task = simple_quadratic(a=1, b=-2, c=1)

    target_space: Space = _space({"x": "uniform(0, 10)"})
    target_task = simple_quadratic(a=1, b=-2, c=1)

    # TODO: Should this max_trials include the number trials from warm starting?
    # My take (@lebrice): Probably not. Doesn't make sense, since the KB is independent from
    # the target experiment, and we shouldn't expect users to know exactly how many trials
    # there are in the KB.
    # IDEA: The MultiTaskWrapper could overwrite the `max_trials` property's setter, so that it
    # remains unset, until we know how many trials we have from other experiments (call it N).
    # Then, once that number is known, the max_trials on the wrapped algorithm could be set to
    # `max_trials + N`.
    # TODO: For now, I'm setting this to a larger number, just to make sure that we get some
    # new trials from the target experiment.
    n_source_trials = 20  # Number of trials from the source task
    max_trials = 40  # Number of trials in the target task

    previous_trials = source_space.sample(n_source_trials)
    previous_trials = [
        add_result(trial, source_task(trial.params["x"])) for trial in previous_trials
    ]
    # Populate the knowledge base.
    knowledge_base = DummyKnowledgeBase(
        [
            (
                ExperimentInfo.from_dict(
                    build_experiment(
                        "source", space=source_space, debug=True
                    ).configuration
                ),
                previous_trials,
            )
        ]
    )

    without_warm_starting = workon(
        _wrap(target_task),
        name="default",
        space=target_space,
        max_trials=max_trials,
        algorithms=algo,
        knowledge_base=None,
    )

    with_warm_starting = workon(
        _wrap(target_task),
        name="warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithms=algo,
        knowledge_base=knowledge_base,
    )

    best_trial_without = _get_best_trial(without_warm_starting)
    best_trial_with = _get_best_trial(with_warm_starting)
    assert best_trial_with.objective is not None
    assert best_trial_without.objective is not None
    assert best_trial_with.objective.value < best_trial_without.objective.value


@pytest.mark.parametrize("algo", [TPE])
def test_warm_start_benchmarking(algo: type[BaseAlgorithm]):
    """Integration test. Compares the performance in the three cases (cold, warm, hot)-starting.

    - Cold-start: Optimize the target task with no prior knowledge (lower bound);
    - Warm-start: Optimize the target task with some prior knowledge from "related" tasks;
    - Hot-start:  Optimize the target task with some prior knowledge from the *same* task (upper
                  bound).
    """
    source_space: Space = _space({"x": "uniform(0, 10)"})
    source_task = simple_quadratic(a=1, b=-2, c=1)
    source_exp = build_experiment(name="source-2", space=source_space, debug=True)

    target_space: Space = _space({"x": "uniform(0, 10)"})
    target_task = simple_quadratic(a=1, b=-2, c=1)

    n_source_trials = 20  # Number of trials from the source task
    max_trials = 40  # Number of trials in the target task

    cold_start_kb = None
    warm_start_kb = create_dummy_kb(
        [source_space], [n_source_trials], task=source_task, prefix="warm"
    )
    hot_start_kb = create_dummy_kb(
        [source_space], [n_source_trials], task=target_task, prefix="warm"
    )

    # Populate the knowledge base.
    # "cold start": Optimize the target task, without any previous knowledge.
    cold_start = workon(
        _wrap(target_task),
        name="cold_start",
        space=target_space,
        max_trials=max_trials,
        algorithms=algo,
        knowledge_base=None,
    )

    warm_start = workon(
        _wrap(target_task),
        name="warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithms=algo,
        knowledge_base=warm_start_kb,
    )

    hot_start = workon(
        _wrap(target_task),
        name="warm_start",
        space=target_space,
        max_trials=max_trials,
        algorithms=algo,
        knowledge_base=hot_start_kb,
    )

    cold_objective = _get_best_trial_objective(cold_start)
    warm_objective = _get_best_trial_objective(warm_start)
    hot_objective = _get_best_trial_objective(hot_start)
    assert cold_objective < warm_objective < hot_objective


P = ParamSpec("P")


def _wrap(objective_fn: Callable[P, float]) -> Callable[P, list[dict]]:
    """Adds some common boilerplate to this objective function."""
    return _with_results(_count_calls(objective_fn))


def _with_results(objective_fn: Callable[P, float]) -> Callable[P, list[dict]]:
    """Adds the boilerplate to create a list of 'Results' dictionary from the objective."""

    @functools.wraps(objective_fn)
    def _with_results(*args: P.args, **kwargs: P.kwargs) -> list[dict]:
        objective = objective_fn(*args, **kwargs)
        return [dict(name="objective", type="objective", value=objective)]

    return _with_results


T = TypeVar("T")


def _count_calls(objective_fn: Callable[P, T]) -> Callable[P, T]:
    count = 0
    signature = inspect.signature(objective_fn, follow_wrapped=True)

    @functools.wraps(objective_fn)
    def _count_calls(*args: P.args, **kwargs: P.kwargs):
        nonlocal count
        count += 1
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        print(f"Starting Trial #{count} with {bound_args.arguments}")
        results = objective_fn(*args, **kwargs)
        print(f"Results: {results}")
        return results

    return _count_calls


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
