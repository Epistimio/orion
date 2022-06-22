from __future__ import annotations

import copy
from typing import Callable

import pytest

from orion.algo.space import Space
from orion.client import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start import KnowledgeBase
from orion.core.worker.warm_start.experiment_config import ExperimentInfo

# Function to create a space.
space: Callable[[dict], Space] = SpaceBuilder().build


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

    def __init__(self, previous_experiments: list[ExperimentInfo]):
        self.previous_experiments = previous_experiments

    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient,
        max_trials: int | None = None,
    ) -> list[tuple[ExperimentInfo, list[Trial]]]:
        """Returns"""
        related_things = []
        for i, experiment_info in enumerate(self.previous_experiments):
            exp_space = space(experiment_info.space)
            previous_trials = exp_space.sample(10)
            previous_trials_with_results = [
                _add_result(trial, i * 100 + j)
                for j, trial in enumerate(previous_trials)
            ]
            related_things.append((experiment_info, previous_trials_with_results))
        return related_things

    def add_experiment(self, experiment: Experiment | ExperimentClient) -> None:
        self.previous_experiments.append(
            ExperimentInfo.from_dict(experiment.configuration)
        )

    @property
    def n_stored_experiments(self) -> int:
        return len(self.previous_experiments)

    def __contains__(self, obj: object) -> bool:
        return obj in self.previous_experiments


@pytest.fixture()
def knowledge_base():
    previous_experiments = []
    return DummyKnowledgeBase(previous_experiments)
