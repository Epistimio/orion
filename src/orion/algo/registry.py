""" Classes that serve as an in-memory storage of trials for the algorithms. """
from __future__ import annotations

import copy
from collections import defaultdict
from logging import getLogger as get_logger
from typing import Any, Container, Iterator, Mapping

from orion.core.worker.trial import Trial, TrialCM

logger = get_logger(__name__)


class Registry(Container[Trial]):
    """In-memory container for the trials that the algorithm suggests/observes/etc."""

    def __init__(self):
        self._trials: dict[str, Trial] = {}

    def __contains__(self, trial_or_id: str | Trial | Any) -> bool:
        if isinstance(trial_or_id, TrialCM):
            trial_id = _get_id(trial_or_id._cm_trial)
        elif isinstance(trial_or_id, Trial):
            trial_id = _get_id(trial_or_id)
        elif isinstance(trial_or_id, str):
            trial_id = trial_or_id
        else:
            raise NotImplementedError(trial_or_id)
        return trial_id in self._trials

    def __getitem__(self, item: int | str) -> Trial:
        if not isinstance(item, (str, int)):
            raise KeyError(item)
        if isinstance(item, str):
            return self._trials[item]
        trial_ids = list(self._trials.keys())
        return self._trials[trial_ids[item]]

    def __iter__(self) -> Iterator[Trial]:
        return iter(self._trials.values())

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def state_dict(self) -> dict:
        return {"_trials": self._trials}

    def set_state(self, statedict: dict) -> None:
        self._trials = statedict["_trials"]

    def has_suggested(self, trial: Trial) -> bool:
        return _get_id(trial) in self

    def has_observed(self, trial: Trial) -> bool:
        trial_id = _get_id(trial)
        if trial_id not in self:
            return False
        return self[trial_id].status in ("broken", "completed")

    def register(self, trial: Trial) -> str:
        """Register the given trial in the registry."""
        trial_id = _get_id(trial)
        if trial_id in self:
            existing = self._trials[trial_id]
            logger.debug(
                f"Overwriting existing trial {existing} with new trial {trial}"
            )
        else:
            logger.debug(
                f"Registry {id(self)} Registering new trial {trial} ({len(self)} trials in total)"
            )
        trial_copy = copy.deepcopy(trial)
        self._trials[trial_id] = trial_copy
        return trial_id


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

    def __iter__(self) -> Iterator[tuple[Trial, list[Trial]]]:
        for trial in self.original_registry:
            yield trial, self[trial]

    def __len__(self) -> int:
        return len(self._mapping)

    def __contains__(self, trial: Trial):
        return trial in self.original_registry

    def __getitem__(self, item: Trial) -> list[Trial]:
        trial_id = _get_id(item)
        if trial_id not in self._mapping:
            if trial_id in self.original_registry:
                return []
            raise KeyError(item)
        transformed_trial_ids = self._mapping[trial_id]
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
        # NOTE: Choosing not to register the trials here, and instead do it more manually.
        # original_id = self.original_registry.register(original_trial)
        # transformed_id = self.transformed_registry.register(transformed_trial)
        original_trial_id = _get_id(original_trial)
        transformed_trial_id = _get_id(transformed_trial)
        self._mapping[original_trial_id].add(transformed_trial_id)
        return original_trial_id


def _get_id(trial: Trial) -> str:
    """Returns the unique identifier to be used to store the trial.

    Only to be used internally in this module. This ignores the `experiment`
    attribute of the trial.
    """
    return Trial.compute_trial_hash(trial, ignore_experiment=True)
