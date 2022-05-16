""" Interface for the Knowledge Base, which is not currently in the Orion codebase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger

from orion.client import ExperimentClient
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.storage.base import BaseStorageProtocol as Storage

log = getLogger(__file__)


# FIXME: Re-introduce this dataclass that contains info about the experiments?
ExperimentInfo = object


class AbstractKnowledgeBase(ABC):
    """Abstract Base Class for the KnowledgeBase, which currently isn't part of the
    Orion codebase.
    """

    @abstractmethod
    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient,
        max_trials: int | None = None,
    ) -> dict[ExperimentInfo, list[Trial]]:
        """Retrieve experiments 'similar' to `target_experiment` and their trials.

        When `max_trials` is given, only up to `max_trials` are returned in total for
        all experiments.

        Parameters
        ----------
        target_experiment : Union[Experiment, ExperimentClient]
            The target experiment, or experiment client.
        max_trials : int, optional
            Maximum total number of trials to fetch. By default `None`, in which case
            all trials from all 'related' experiments are returned.

        Returns
        -------
        Dict[ExperimentInfo, List[Trial]]
            Dictionary mapping from `ExperimentInfo` objects to a list of trials from
            that experiment.
        """

    @property
    @abstractmethod
    def n_stored_experiments(self) -> int:
        """Returns the current number of experiments registered the Knowledge base."""

    @abstractmethod
    def add_storage(self, storage: Storage) -> None:
        """Adds the experiments from the given `storage`.

        Parameters
        ----------
        storage : Storage
            Storage object.
        """

    @abstractmethod
    def add_experiment(self, experiment: Experiment | ExperimentClient) -> None:
        """Adds trials from the given experiment to the knowledge base.

        Parameters
        ----------
        experiment : Union[Experiment, ExperimentClient]
            Experiment or experiment client to add to the KB.
        """
