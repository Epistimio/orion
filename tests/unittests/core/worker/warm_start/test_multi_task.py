""" Tests for the multi-task wrapper that is applied to regular algorithms when using a KB."""
# pylint: disable=protected-access,invalid-name
from __future__ import annotations

import copy
import random
from typing import Any, Callable, Sequence

import pytest

from orion.algo.base import BaseAlgorithm
from orion.algo.gridsearch import GridSearch
from orion.algo.space import Space
from orion.algo.tpe import TPE
from orion.client import build_experiment
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.algo_wrappers.insist_suggest import InsistSuggest
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.experiment_config import ExperimentInfo
from orion.core.worker.warm_start.knowledge_base import KnowledgeBase
from orion.core.worker.warm_start.multi_task_wrapper import MultiTaskWrapper
from orion.testing.dummy_algo import FixedSuggestionAlgo

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
) -> DummyKnowledgeBase:
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

    @pytest.mark.parametrize("n_previous_experiments", [1, 2])
    def test_collisions(self, n_previous_experiments: int):
        """Test for the collisions:

        The wrapped algo suggests 2 trials, with different task ids, but the same other parameters:
        {"task_id": 0, "x": 1}, {"task_id": 1, "x": 1}
        The wrapper should then suggest a single trial, since it removes the task id:
        {"x": 1}
        Then, when the wrapper observes a result for that trial:
        {"x": 1} (objective = 123)
        The wrapped algo should then have both the original trials set to have that new result:
        {"task_id": 0, "x": 1} (objective=123), {"task_id": 1, "x": 1} (objective=123)
        """
        # NOTE: When n_previous_experiments is 1, the `task_id` dimension is a binary logit
        # But when n_previous_experiments is >=2, the `task_id` dimension becomes categorical and
        # is split into task_id[0], task_id[1], ..., task_id[n_previous_experiments].
        kb = create_dummy_kb(
            previous_spaces=previous_spaces[:n_previous_experiments],
            n_trials_per_space=1,  # doesn't matter for this test.
        )
        wrapper = create_algo(
            FixedSuggestionAlgo,
            space=target_space,
            knowledge_base=kb,
            seed=123,
        )
        algo = wrapper.unwrapped
        assert isinstance(wrapper, MultiTaskWrapper)
        assert isinstance(algo, FixedSuggestionAlgo)

        assert algo.fixed_suggestion in algo.space

        # Create two trials with same params but different task id.
        t1 = algo.fixed_suggestion
        t2 = copy.deepcopy(algo.fixed_suggestion)
        if n_previous_experiments == 1:
            assert "task_id" in t1.params
            # The task_id should be set to 0, since we set the prior of the space to only allow
            # sampling trials with task_id of 0 (see other test).
            assert t1.params["task_id"] == 0

            # Set t2 to have the other task id.
            # TODO:: Can't change a trial's params like this. The `params` property is just a view.
            # t2.params["task_id"] = 1.0
            _set_params(t2, {"task_id": 1})
        else:
            # Same here, only the task_id[0] should have a non-zero value (of 1).
            assert t1.params["task_id[0]"] == 1
            assert all(
                t1.params[f"task_id[{i}]"] == 0
                for i in range(1, n_previous_experiments + 1)
            )
            _set_params(t2, {"task_id[0]": 0.0, "task_id[1]": 1.0})

        assert t1.params != t2.params
        assert t1.id != t2.id

        # Get the transformed version of t1 from the Wrapper.
        wrapper_t = wrapper.suggest(1)[0]

        algo.fixed_suggestion = t2
        new_wrapper_suggestions = wrapper.suggest(1)
        # The wrapper can't suggest anything, because the trial is a duplicate of the previous when
        # the task ids are removed.
        assert not new_wrapper_suggestions

        # NOTE: The collision should be recorded in this MultiTask wrapper, not in the
        # SpaceTransform wrapper!
        assert wrapper_t in wrapper.registry_mapping
        equivalent_trials = wrapper.registry_mapping[wrapper_t]
        assert len(equivalent_trials) == 2

        assert {t.params["task_id"] for t in equivalent_trials} == {0, 1}

    @pytest.mark.parametrize("n_previous_experiments", [1, 2])
    @pytest.mark.parametrize("algo_type", [Random, TPE, GridSearch])
    def test_algo_suggests_task0(
        self, n_previous_experiments: int, algo_type: type[BaseAlgorithm]
    ):
        """Test that the wrapped algo space only produces trials with task_id of 0 when sampled."""
        # NOTE: When n_previous_experiments is 1, the `task_id` dimension is a binary logit
        # But when n_previous_experiments is >=2, the `task_id` dimension becomes categorical and
        # is split into task_id[0], task_id[1], ..., task_id[n_previous_experiments].
        kb = create_dummy_kb(
            previous_spaces=previous_spaces[:n_previous_experiments],
            n_trials_per_space=1,  # doesn't matter for this test.
        )
        wrapper = create_algo(
            algo_type,
            space=target_space,
            knowledge_base=kb,
        )
        wrapper.max_trials = 100
        wrapper.suggest(10)
        # NOTE: It doesn't really matter what type of wrapper `wrapper.algorithm` is here.
        # It is the first wrapper in the chain that sees the task ids.
        assert len(wrapper.algorithm.registry) >= 10
        assert all(_get_task_id(trial) == 0 for trial in wrapper.algorithm.registry)

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

        algo = create_algo(
            Random, space=target_space, knowledge_base=knowledge_base, seed=123
        )
        algo.max_trials = 10
        assert algo.n_suggested == 0
        assert algo.n_observed == 0
        assert not algo.is_done
        assert not algo.unwrapped.is_done

        algo.warm_start(knowledge_base.related_trials)
        trials_in_kb = sum(len(trials) for _, trials in knowledge_base.related_trials)
        assert algo.unwrapped.n_observed == trials_in_kb
        assert not algo.is_done
        assert not algo.unwrapped.is_done

    def test_setting_max_trials(self):
        """Test that the value of the max_trials property is increased by the MultiTaskWrapper by
        the number of trials observed in warm-starting before it is passed down to the algo.
        """
        max_trials = 5
        previous_trials = 10
        knowledge_base = create_dummy_kb(
            previous_spaces=[target_space], n_trials_per_space=previous_trials
        )
        algo = create_algo(Random, space=target_space, knowledge_base=knowledge_base)
        algo.max_trials = max_trials
        assert algo.unwrapped.max_trials is None
        assert algo.max_trials == max_trials
        algo.warm_start(knowledge_base.related_trials)
        assert algo.unwrapped.max_trials == previous_trials + max_trials
        assert algo.max_trials == max_trials

    def test_n_observed_n_suggested(self):
        """Test that the n_observed and n_suggested aren't affected by the warm-starting."""
        previous_trials = 10
        knowledge_base = create_dummy_kb(
            previous_spaces=[target_space], n_trials_per_space=previous_trials
        )
        algo = create_algo(Random, space=target_space, knowledge_base=knowledge_base)
        assert algo.n_observed == 0
        assert algo.n_suggested == 0
        algo.warm_start(knowledge_base.related_trials)
        assert algo.n_observed == 0
        assert algo.n_suggested == 0
        assert algo.unwrapped.n_observed == previous_trials


def _set_params(trial: Trial, params: dict[str, Any]) -> None:
    # TODO: It's really hard to set a new value for a hyperparameter in a trial object.
    for name, value in params.items():
        param_object_index = [p.name == name for p in trial._params].index(True)
        trial._params[param_object_index].value = value


def _get_task_id(trial: Trial) -> int:
    if "task_id" in trial.params:
        return trial.params["task_id"]
    n_tasks = sum(
        "task_id[" in dimension_name for dimension_name in trial.params.keys()
    )
    values = [trial.params[f"task_id[{i}]"] for i in range(n_tasks)]
    return values.index(max(values))
