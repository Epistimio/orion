""" Classes that serve as an in-memory storage of trials for the algorithms. """
from __future__ import annotations

import copy
from collections import defaultdict
from logging import getLogger as get_logger
from typing import Any, Container, Iterable, Iterator, Mapping

from orion.core.worker.trial import Trial, TrialCM

logger = get_logger(__name__)


class Registry(Container[Trial]):
    """In-memory container for the trials that the algorithm suggests/observes/etc.

    This behaves a bit like a managed dictionary, but the "keys" are trials ids, which
    (at the time of writing) can vary depending on how we chose to compute them.
    """

    def __init__(self, trials: Iterable[Trial] = ()):
        self._trials: dict[str, Trial] = {}
        for trial in trials:
            self.register(trial)

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}({list(iter(self))})"

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

    def __getitem__(self, item: str) -> Trial:
        if not isinstance(item, str):
            raise KeyError(item)
        return self._trials[item]

    def __iter__(self) -> Iterator[Trial]:
        return iter(self._trials.values())

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def state_dict(self) -> dict:
        """Get the state of the registry as a dictionary."""
        return {"_trials": copy.deepcopy(self._trials)}

    def set_state(self, statedict: dict) -> None:
        """Set the state of the registry from the given dictionary."""
        self._trials = copy.deepcopy(statedict["_trials"])

    def has_suggested(self, trial: Trial) -> bool:
        """Check if the trial has been suggested."""
        return _get_id(trial) in self

    def has_observed(self, trial: Trial) -> bool:
        """Check if the trial has been observed."""
        trial_id = _get_id(trial)
        if trial_id not in self:
            return False
        return self[trial_id].status in ("broken", "completed")

    def register(self, trial: Trial) -> str:
        """Register the given trial in the registry."""
        trial_id = _get_id(trial)
        if trial_id in self:
            existing = self._trials[trial_id]
            if existing.status != "new" and trial.status == "new":
                raise RuntimeError(
                    f"Can't overwrite existing (older) trial {existing} with new trial {trial}!"
                )
            logger.debug("Overwriting existing trial %s with %s", existing, trial)
        else:
            logger.debug(
                "Registry %s Registering new trial %s (%s trials in total)",
                id(self),
                trial,
                len(self),
            )
        trial_copy = copy.deepcopy(trial)
        self._trials[trial_id] = trial_copy
        return trial_id

    def get_existing(self, trial: Trial) -> Trial:
        """Get the equivalent trial from the registry.

        If `trial` isn't in the registry, raises a RuntimeError.
        """
        trial_id = _get_id(trial)
        if trial_id not in self:
            raise RuntimeError(f"Trial `{trial}` isn't in the registry (id={trial_id})")
        return self[trial_id]


class RegistryMapping(Mapping[Trial, "list[Trial]"]):
    """A map between the original and transformed registries.

    This object is used in the `SpaceTransform` to check if a trial in the original space
    has equivalent trials in the transformed space.

    The goal is to make it so the algorithms don't have to care about the transforms/etc.
    """

    def __init__(self, original_registry: Registry, transformed_registry: Registry):
        self.original_registry = original_registry
        self.transformed_registry = transformed_registry
        self._mapping: dict[str, set[str]] = defaultdict(set)

    @property
    def state_dict(self) -> dict:
        """Get the state of the registry mapping as a dictionary.

        NOTE: This does NOT include the state of the individual registries.
        """
        return {
            "_mapping": copy.deepcopy(self._mapping),
        }

    def set_state(self, statedict: dict):
        """Set the state of the registry mapping from the given dictionary.

        NOTE: This does NOT set the state of the individual registries.
        """
        self._mapping = copy.deepcopy(statedict["_mapping"])

    def __iter__(self) -> Iterator[Trial]:
        """Iterate over the trials in the original registry."""
        for trial_id in self._mapping:
            yield self.original_registry[trial_id]

    def __len__(self) -> int:
        """Give the number of trials in the mapping.

        This should be the same as the number of trials in the original registry.
        """
        return len(self._mapping)

    def __contains__(self, trial: Trial):
        return _get_id(trial) in self._mapping

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

    def __repr__(self) -> str:
        return (
            f"{type(self).__qualname__}"
            f"({list((trial, self.get_trials(trial)) for trial in self)})"
        )


def _get_id(trial: Trial) -> str:
    """Returns the unique identifier to be used to store the trial.

    Only to be used internally in this module. This ignores the `experiment`
    attribute of the trial.
    """
    return Trial.compute_trial_hash(trial)
