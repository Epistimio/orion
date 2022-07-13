""" Interface for the Knowledge Base, which is not currently in the Orion codebase.
"""
from __future__ import annotations

import inspect
import typing
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Container

if typing.TYPE_CHECKING:
    from orion.client import ExperimentClient
    from orion.core.worker.experiment import Experiment
    from orion.core.worker.trial import Trial
    from orion.storage.base import BaseStorageProtocol


log = getLogger(__file__)

from .experiment_config import ExperimentInfo


class KnowledgeBase(ABC, Container[ExperimentInfo]):
    """Abstract Base Class for the KnowledgeBase, which currently isn't part of the
    Orion codebase.
    """

    @abstractmethod
    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient,
        max_trials: int | None = None,
    ) -> list[tuple[ExperimentInfo, list[Trial]]]:
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
        list[tuple[ExperimentInfo, list[Trial]]]
            A list of tuples of experiments and the similar trials from that experiment.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_stored_experiments(self) -> int:
        """Returns the current number of experiments registered the Knowledge base."""

    @abstractmethod
    def add_experiment(self, experiment: Experiment | ExperimentClient) -> None:
        """Adds trials from the given experiment to the knowledge base.

        Parameters
        ----------
        experiment : Union[Experiment, ExperimentClient]
            Experiment or experiment client to add to the KB.
        """

    @property
    def configuration(self) -> dict[str, Any]:
        """Returns the configuration of the knowledge base.

        By default, returns a dictionary containing the attributes of `self` which are also
        constructor arguments.
        """
        init_signature = inspect.signature(type(self).__init__)
        init_arguments_attributes = {
            name: getattr(self, name)
            for name in init_signature.parameters
            if name != "self" and hasattr(self, name)
        }
        return {type(self).__qualname__: init_arguments_attributes}

    # NOTE: Not making this an abstract method, since we might need to adapt this a bit.
    # @abstractmethod
    def add_storage(self, storage: BaseStorageProtocol) -> None:
        """Adds the experiments from the given `storage`.

        Parameters
        ----------
        storage : Storage
            Storage object.
        """
        raise NotImplementedError
