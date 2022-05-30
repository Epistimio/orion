from __future__ import annotations

import copy
import pytest

from orion.algo.space import Space
from orion.core.worker.warm_start import KnowledgeBase
from typing import Any, Callable
from orion.core.io.space_builder import SpaceBuilder
from orion.client import ExperimentClient
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.testing.state import OrionState

# Function to create a space.
space: Callable[[dict], Space] = SpaceBuilder().build

previous_spaces: list[Space] = [
    space({"x": "uniform(0, 5)"}),
    space({"x": "uniform(1, 6)"}),
    space({"x": "uniform(2, 7)"}),
    space({"x": "uniform(3, 8)"}),
    space({"x": "uniform(4, 9)"}),
]
target_space = space({"x": "uniform(0, 10)"})


def fake_experiment(
    name: str, space: Space, trials: list[Trial] | None = None
) -> Experiment:
    """Creates a fake experiment with the given trials."""
    with OrionState(trials=trials):
        # TODO: Use an ExperimentInfo here instead?
        experiment = Experiment(name=name, space=space)
        return experiment


# TODO: Use an ExperimentInfo here instead?
previous_experiments: list[Experiment] = [
    fake_experiment(name=f"fake_{i}", space=space_i, trials=space_i.sample(10))
    for i, space_i in enumerate(previous_spaces)
]


def _add_result(trial: Trial, objective: float) -> Trial:
    """Add a random result to the trial."""
    new_trial = copy.deepcopy(trial)
    new_trial.status = "completed"
    new_trial.results.append(
        Trial.Result(name="objective", type="objective", value=objective)
    )
    return new_trial


class DummyKnowledgeBase(KnowledgeBase):
    """Knowledge base that returns fake trials from fake "similar" experiments.

    For the moment, we define "similarity" between experiments purely based on their search space.
    (similar spaces)
    """

    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient,
        max_trials: int | None = None,
    ) -> list[tuple[Experiment, list[Trial]]]:
        """Returns"""
        assert target_experiment.space == target_space
        return [
            (
                experiment_i,
                [
                    _add_result(trial, i * 100 + j)
                    for j, trial in experiment_i.space.sample(10)
                ],
            )
            for i, experiment_i in enumerate(previous_experiments)
        ]


@pytest.fixture()
def knowledge_base():
    pass


class TestMultiTaskWrapper:
    """Tests for the multi-task wrapper."""

    def test_adds_task_id(self):
        """Test that when an algo is wrapped with the multi-task wrapper, the trials it returns
        with suggest() have an additional 'task-id' value.
        """
