""" Tests for the multi-task wrapper that is applied to regular algorithms when using a KB."""
from __future__ import annotations

import functools
import inspect
import random
from typing import Callable, Sequence

import pytest
from typing_extensions import ParamSpec

from orion.algo.space import Space
from orion.client import build_experiment, workon
from orion.client.experiment import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.algo_wrappers.insist_suggest import InsistSuggest
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.experiment_config import ExperimentInfo
from orion.core.worker.warm_start.knowledge_base import KnowledgeBase
from orion.core.worker.warm_start.multi_task_wrapper import MultiTaskWrapper

from .test_knowledge_base import DummyKnowledgeBase, add_result

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


@pytest.fixture()
def knowledge_base() -> KnowledgeBase:
    """Dummy knowledge base fixture used for testing."""
    # The KB gets populated with the previous experiments.
    previous_experiments = [
        ExperimentInfo.from_dict(
            build_experiment(name=f"foo-{i}", space=space_i, debug=True).configuration
        )
        for i, space_i in enumerate(previous_spaces)
    ]
    return DummyKnowledgeBase(previous_experiments)


from orion.algo.random import Random
from orion.core.worker.warm_start import WarmStarteable


class DummyWarmStarteableAlgo(Random, WarmStarteable):
    """A dummy warm-starteable algorithm.

    Saves the warm start trials in an attribute for later inspection.
    """

    def __init__(self, space: Space, seed: int | Sequence[int] | None = None):
        super().__init__(space, seed)
        self._warm_start_trials = {}

    def warm_start(self, warm_start_trials: dict[ExperimentInfo, list[Trial]]):
        self._warm_start_trials = warm_start_trials

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}()"


class TestCreateAlgo:
    """Tests for the `create_algo` function. Checks that the right wrappers are applied."""

    def test_wrappers_are_applied(self, knowledge_base: KnowledgeBase):
        """Test that the MultiTaskWrapper is added whenever a knowledge base is passed and the
        algorithm is not warm-starteable.
        """
        # No knowledge base, not warm-starteable.
        algo = create_algo(Random, space=target_space)
        assert isinstance(algo, InsistSuggest)
        assert isinstance(algo.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm, Random)
        assert algo.unwrapped is algo.algorithm.algorithm

        # No knowledge base, warm-starteable.
        algo = create_algo(DummyWarmStarteableAlgo, space=target_space)
        assert isinstance(algo, InsistSuggest)
        assert isinstance(algo.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm, DummyWarmStarteableAlgo)
        assert algo.unwrapped is algo.algorithm.algorithm

        # With a Knowledge base + regular (non-warm-starteable) algo.
        algo = create_algo(Random, space=target_space, knowledge_base=knowledge_base)
        assert isinstance(algo, MultiTaskWrapper)
        assert isinstance(algo.algorithm, InsistSuggest)
        assert isinstance(algo.algorithm.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm.algorithm, Random)
        assert algo.unwrapped is algo.algorithm.algorithm.algorithm

        # With a Knowledge base + warm-starteable algo.
        algo = create_algo(
            DummyWarmStarteableAlgo, space=target_space, knowledge_base=knowledge_base
        )
        assert isinstance(algo, InsistSuggest)
        assert isinstance(algo.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm, Random)
        assert algo.unwrapped is algo.algorithm.algorithm


class TestMultiTaskWrapper:
    """Tests for the multi-task wrapper."""

    def test_adds_task_id(self, knowledge_base: KnowledgeBase):
        """Test that when an algo is wrapped with the multi-task wrapper, the trials it returns
        with suggest() have an additional 'task-id' value.
        """
        algo = create_algo(DummyWarmStarteableAlgo, space=target_space)

        experiment = build_experiment(
            name="target",
            space=target_space,
            knowledge_base=knowledge_base,
            algorithms=DummyWarmStarteableAlgo,
            max_trials=100,
            debug=True,
        )
        while not experiment.is_done:
            trial = experiment.suggest()
            objective = random.random()
            trial_with_result = add_result(trial, objective)
            # TODO: Fix outdated format of `experiment.observe` method.
            experiment.observe(trial_with_result, [])
        assert experiment._experiment.knowledge_base is knowledge_base
        algo = experiment.algorithms
        # TODO: Add type hints to the `Experiment` class, and make it generic in terms of the
        # algorithms, so that we could get the `algorithms` to be `Random` here.
        assert isinstance(algo, InsistSuggest)
        assert isinstance(algo.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm, DummyWarmStarteableAlgo)

        unwrapped_algo = algo.unwrapped
        assert isinstance(unwrapped_algo, DummyWarmStarteableAlgo)
        assert unwrapped_algo._warm_start_trials


def test_warm_starting_helps(knowledge_base: KnowledgeBase):
    """Integration test. Shows that warm-starting helps in a simple task."""
    target_space: Space = _space({"x": "uniform(0, 10)"})

    # TODO
    # experiment = build_experiment(
    #     name="target",
    #     space=target_space,
    #     knowledge_base=knowledge_base,
    #     algorithms=algo,
    #     max_trials=100,
    #     debug=True,
    # )

    from orion.algo.tpe import TPE

    P = ParamSpec("P")

    def with_results(objective_fn: Callable[P, float]) -> Callable[P, list[dict]]:
        @functools.wraps(objective_fn)
        def _with_results(*args: P.args, **kwargs: P.kwargs) -> list[dict]:
            objective = objective_fn(*args, **kwargs)
            return [dict(name="objective", type="objective", value=objective)]

        return _with_results

    def count_calls(objective_fn):
        count = 0
        signature = inspect.signature(objective_fn, follow_wrapped=True)

        @functools.wraps(objective_fn)
        def _count_calls(*args, **kwargs):
            nonlocal count
            count += 1
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            print(f"Starting Trial #{count} with {bound_args.arguments}")
            results = objective_fn(*args, **kwargs)
            print(f"Results: {results}")
            return results

        return _count_calls

    def simple_quadratic(x: float) -> float:
        return x**2 - 2 * x + 1

    target_fn = count_calls(with_results(simple_quadratic))

    cold_start = workon(
        target_fn,
        name="cold_start",
        space=target_space,
        max_trials=20,
        algorithms=TPE,
        knowledge_base=None,
    )

    def _get_objective(trial: Trial) -> float:
        objective: Trial.Result | None = trial.objective
        if objective is not None:
            return objective.value
        return float("inf")

    def _get_best_trial(experiment: ExperimentClient) -> Trial:
        trials: list[Trial] = experiment.fetch_trials_by_status(status="completed")
        return min(trials, key=_get_objective)

    cold_start_best_trial = _get_best_trial(cold_start)

    # Now, add the first experiment to the knowledge base.
    knowledge_base.add_experiment(cold_start)
    hot_start = workon(
        target_fn,
        name="hot_start",
        space=target_space,
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
        max_trials=40,
        algorithms=TPE,
        knowledge_base=knowledge_base,
    )
    hot_start_best_trial = _get_best_trial(hot_start)
    assert hot_start_best_trial.objective is not None
    assert cold_start_best_trial.objective is not None
    assert hot_start_best_trial.objective.value < cold_start_best_trial.objective.value
