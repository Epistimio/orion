""" Tests for the multi-task wrapper that is applied to regular algorithms when using a KB."""
# pylint: disable=protected-access
from __future__ import annotations

import functools
import inspect
import random
from typing import Callable, Sequence, TypeVar

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
    experiments = [
        build_experiment(name=f"foo-{i}", space=space_i, debug=True)
        for i, space_i in enumerate(previous_spaces)
    ]
    previous_trials = [
        (
            ExperimentInfo.from_dict(experiment.configuration),
            [add_result(t, i) for i, t in enumerate(experiment.space.sample(10))],
        )
        for experiment in experiments
    ]
    return DummyKnowledgeBase(previous_trials)


def create_dummy_kb(
    previous_spaces: Sequence[Space],
    n_trials_per_space: int | Sequence[int] = 10,
    task: Callable[..., float] | None = None,
    prefix: str = "foo",
) -> KnowledgeBase:
    """Create a KB with the experiments/trials that we want."""
    if isinstance(n_trials_per_space, int):
        n_trials_per_space = [n_trials_per_space for _ in previous_spaces]
    else:
        assert len(n_trials_per_space) == len(previous_spaces)
    experiments = [
        build_experiment(name=f"{prefix}-{i}", space=space_i, debug=True)
        for i, space_i in enumerate(previous_spaces)
    ]
    previous_trials = [
        (
            ExperimentInfo.from_dict(experiment.configuration),
            [
                add_result(trial, (task(**trial.params) if task else j))
                for j, trial in enumerate(experiment.space.sample(n_trials))
            ],
        )
        for experiment, n_trials in zip(experiments, n_trials_per_space)
    ]
    return DummyKnowledgeBase(previous_trials)


from orion.algo.random import Random
from orion.core.worker.warm_start import WarmStarteable


class DummyWarmStarteableAlgo(Random, WarmStarteable):
    """A dummy warm-starteable algorithm.

    Saves the warm start trials in an attribute for later inspection.
    """

    def __init__(self, space: Space, seed: int | Sequence[int] | None = None):
        super().__init__(space, seed)
        self.warm_start_trials: list[tuple[ExperimentInfo, list[Trial]]] = []

    def warm_start(self, warm_start_trials: list[tuple[ExperimentInfo, list[Trial]]]):
        self.warm_start_trials = warm_start_trials

    def observe(self, trials: list[Trial]) -> None:
        return super().observe(trials)

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

    def test_collisions(self):
        """TODO: Write a test for the collisions:

        The wrapped algo suggests 2 trials, with different task ids, but the same other parameters:
        {"task_id": 0, "x": 1}, {"task_id": 1, "x": 1}
        The wrapper should then suggest a single trial, since it removes the task id:
        {"x": 1}
        Then, when the wrapper observes a result for that trial:
        {"x": 1} (objective = 123)
        The wrapped algo should then have both the original trials set to have that new result:
        {"task_id": 0, "x": 1} (objective=123), {"task_id": 1, "x": 1} (objective=123)
        """
        raise NotImplementedError

    def test_wrapped_algo_suggests_task_id_0(self):
        """Test that the wrapped algo space only produces trials with task_id of 0 when sampled."""
        raise NotImplementedError

    def test_adds_task_id(self, knowledge_base: KnowledgeBase):
        """Test that when an algo is wrapped with the multi-task wrapper, the trials it returns
        with suggest() have an additional 'task-id' value.
        """
        algo_type = Random
        max_trials = 10
        experiment = build_experiment(
            name="target_1",
            space=target_space,
            knowledge_base=knowledge_base,
            algorithms=algo_type,
            max_trials=max_trials,
            debug=True,
        )
        assert experiment._experiment.knowledge_base is knowledge_base
        # IDEA: Add type hints to the `Experiment` class, and make it generic in terms of the
        # algorithms, so that we could get the `algorithms` to be algo_type here.
        algo = experiment.algorithms
        assert isinstance(algo, MultiTaskWrapper)
        assert isinstance(algo.unwrapped, algo_type)
        assert experiment.fetch_trials() == []
        assert not experiment.is_done

        trial = algo.suggest(1)[0]
        assert len(algo.registry) == 1
        assert list(algo.registry) == [trial]
        # There may be more than one trial which map to this one.
        assert len(algo.unwrapped.registry) >= 1
        assert all("task_id" in trial.params for trial in algo.unwrapped.registry)

        trial_with_result = add_result(trial, 0)
        algo.observe([trial_with_result])
        assert all("task_id" in trial.params for trial in algo.unwrapped.registry)

    def test_with_warmstarteable_algo(self, knowledge_base: DummyKnowledgeBase):
        """Test that when using a warm-starteable algorithm, the multi-task wrapper is not used,
        and the `warm_start` method receives all the related trials from the KB.
        """
        experiment = build_experiment(
            name="target_2",
            space=target_space,
            knowledge_base=knowledge_base,
            algorithms=DummyWarmStarteableAlgo,
            max_trials=100,
            debug=True,
        )
        assert experiment._experiment.knowledge_base is knowledge_base
        algo = experiment.algorithms
        assert algo is not None
        assert isinstance(algo.unwrapped, DummyWarmStarteableAlgo)
        assert not algo.unwrapped.warm_start_trials

        while not experiment.is_done:
            trial = experiment.suggest()
            objective = random.random()
            trial_with_result = add_result(trial, objective)
            experiment.observe(trial_with_result, [])

        algo = experiment.algorithms

        # IDEA: Add type hints to the `Experiment` class, and make it generic in terms of the
        # algorithms, so that we could get the `algorithms` to be `Random` here.

        assert isinstance(algo, InsistSuggest)
        assert isinstance(algo.algorithm, SpaceTransform)
        assert isinstance(algo.algorithm.algorithm, DummyWarmStarteableAlgo)

        unwrapped_algo = algo.unwrapped
        assert isinstance(unwrapped_algo, DummyWarmStarteableAlgo)
        # Check that the algo received the transformed trials from the KB.
        assert len(unwrapped_algo.warm_start_trials) == len(
            knowledge_base.related_trials
        )
        for (algo_exp_config, algo_trials), (kb_exp_config, kb_trials) in zip(
            unwrapped_algo.warm_start_trials, knowledge_base.related_trials
        ):
            assert algo_exp_config == kb_exp_config
            # NOTE: Trials aren't identical, because some transformations might have been done
            # on the trials from the KB before the algo observes them.
            # assert algo_trials == kb_trials
            assert len(algo_trials) == len(kb_trials)

    def test_is_done_doesnt_count_trials_from_kb(
        self, knowledge_base: DummyKnowledgeBase
    ):
        """Test that the trials in the Knowledge base don't affect the 'is_done' of the algo."""

        algo = create_algo(Random, space=target_space, knowledge_base=knowledge_base)
        algo.max_trials = 10
        assert algo.n_suggested == 0
        assert algo.n_observed == 0
        assert not algo.is_done
        assert not algo.unwrapped.is_done

        algo.warm_start(knowledge_base.related_trials)
        trials_in_kb = sum(len(trials) for _, trials in knowledge_base.related_trials)
        assert algo.n_suggested == trials_in_kb
        assert algo.n_observed == trials_in_kb
        assert not algo.is_done
        assert not algo.unwrapped.is_done


def simple_quadratic(
    a: float = 1, b: float = -2, c: float = 1
) -> Callable[[float], float]:
    return lambda x: a * x**2 + b * x + c


from orion.algo.base import BaseAlgorithm
from orion.algo.tpe import TPE


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
