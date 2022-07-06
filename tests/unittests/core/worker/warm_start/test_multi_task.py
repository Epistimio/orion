""" Tests for the multi-task wrapper that is applied to regular algorithms when using a KB."""
from __future__ import annotations

import random
from typing import Callable

import pytest

from orion.algo.space import Space
from orion.client import build_experiment
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
def knowledge_base():
    previous_experiments = [
        # build_experiment(name=f"foo-{i}", space=space_i, debug=True)
        ExperimentInfo.from_dict(
            build_experiment(name=f"foo-{i}", space=space_i, debug=True).configuration
        )
        for i, space_i in enumerate(previous_spaces)
    ]
    return DummyKnowledgeBase(previous_experiments)


from orion.algo.random import Random
from orion.core.worker.warm_start import WarmStarteable


class DummyWarmStarteableAlgo(Random, WarmStarteable):
    def __init__(self, space, seed=None):
        super().__init__(space, seed)
        self._warm_start_trials = {}

    def warm_start(self, warm_start_trials: dict[ExperimentInfo, list[Trial]]):
        self._warm_start_trials = warm_start_trials

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}()"


class TestCreateAlgo:
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
