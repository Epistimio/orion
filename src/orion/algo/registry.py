""" Classes that serve as an in-memory storage of trials for the algorithms. """
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterator, Mapping

from orion.core.worker.trial import Trial


class Registry(Mapping[str, Trial]):
    """In-memory container for the trials that the algorithm suggests/observes/etc."""

    def __init__(self):
        self._trials: dict[str, Trial] = {}

    def __contains__(self, trial_or_id: str | Trial | Any) -> bool:
        if isinstance(trial_or_id, Trial):
            trial_id = trial_or_id.id
        elif isinstance(trial_or_id, str):
            trial_id = trial_or_id
        else:
            raise NotImplementedError(trial_or_id)
        return trial_id in self._trials

    def __getitem__(self, item: str) -> Trial:
        if not isinstance(item, str):
            raise KeyError(item)
        return self._trials[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def state_dict(self) -> dict:
        return {"_trials": self._trials}

    def set_state(self, statedict: dict) -> None:
        self._trials = statedict["_trials"]

    def has_suggested(self, trial: Trial) -> bool:
        return trial.id in self

    def has_observed(self, trial: Trial) -> bool:
        if trial.id not in self._trials:
            return False
        return self[trial.id].status in ("broken", "completed")

    def get_trial(self, trial_id: str) -> Trial:
        return self[trial_id]

    def register(self, trial: Trial) -> str:
        """Register the given trial in the registry."""
        if trial.id in self:
            existing = self._trials[trial.id]
            # TODO: Do we allow overwriting? Or maybe only if the status is updated?
            # raise RuntimeError(
            #     f"Trial {trial} is already registered as {self._trials[trial.id]}"
            # )
        trial = copy.deepcopy(trial)
        self._trials[trial.id] = trial
        return trial.id


class RegistryMapping(Mapping[Trial, "list[Trial]"]):
    """A map between the original and transformed registries.

    This object is used in the `SpaceTransformAlgoWrapper` to check if a trial in the original space
    has equivalent trials in the transformed space.

    The goal is to make it so the algorithms don't have to care about the transforms/etc.
    """

    def __init__(self, original_registry: Registry, transformed_registry: Registry):
        self.original_registry = original_registry
        self.transformed_registry = transformed_registry
        self._mapping: dict[str, set[str]] = defaultdict(set)

    @property
    def state_dict(self) -> dict:
        return {
            "original_registry": self.original_registry.state_dict,
            "transformed_registry": self.transformed_registry.state_dict,
            "_mapping": self._mapping,
        }

    def set_state(self, statedict: dict):
        self.original_registry.set_state(statedict["original_registry"])
        self.transformed_registry.set_state(statedict["transformed_registry"])
        self._mapping = statedict["_mapping"]

    def __iter__(self) -> Iterator[Trial]:
        return iter(self.original_registry.values())

    def __len__(self) -> int:
        return len(self.original_registry)

    def __contains__(self, trial: Trial):
        return trial in self.original_registry

    def __getitem__(self, item: Trial) -> list[Trial]:
        if item.id not in self._mapping:
            raise KeyError(item)
        transformed_trial_ids = self._mapping[item.id]
        return [
            self.transformed_registry[transformed_id]
            for transformed_id in transformed_trial_ids
        ]

    def get_trials(self, original_trial: Trial) -> list[Trial]:
        """Return the registered transformed trials that map to the given trial in the original
        space.
        """
        return self.get(original_trial, [])

    def register(self, original_trial: Trial, transformed_trial: Trial) -> str:
        """Register an equivalence between the given original trial and the transformed trial."""
        # NOTE: Peut-être pas .id, faut voir comment on compute le ID.
        original_id = self.original_registry.register(original_trial)
        transformed_id = self.transformed_registry.register(transformed_trial)
        self._mapping[original_id].add(transformed_id)
        return original_id
