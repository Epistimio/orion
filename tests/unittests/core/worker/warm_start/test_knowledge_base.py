from __future__ import annotations

import copy
from typing import Callable

import pytest

from orion.algo.space import Space
from orion.client import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.experiment import Experiment
from orion.core.worker.experiment_config import ExperimentConfig
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start import KnowledgeBase

# Function to create a space.
space: Callable[[dict], Space] = SpaceBuilder().build


def add_result(trial: Trial, objective: float) -> Trial:
    """Add `objective` as the result of `trial`. Returns a new Trial object."""
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

    def __init__(
        self,
        related_trials: list[tuple[ExperimentConfig, list[Trial]]] | None = None,
    ):
        self.related_trials = related_trials or []

    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient,
        max_trials: int | None = None,
    ) -> list[tuple[ExperimentConfig, list[Trial]]]:
        """Returns"""
        return copy.deepcopy(self.related_trials)

    def add_experiment(self, experiment: Experiment | ExperimentClient) -> None:
        self.related_trials.append(
            (
                experiment.configuration,
                experiment.fetch_trials(),
            )
        )

    @property
    def n_stored_experiments(self) -> int:
        return len(self.related_trials)

    def __contains__(self, obj: object) -> bool:
        return obj in self.related_trials


@pytest.fixture()
def knowledge_base():
    return DummyKnowledgeBase()
