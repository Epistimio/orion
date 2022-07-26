""" Knowledge-Base, containing previous trials, which are used to warm-start the HPO algorithm. """
from __future__ import annotations

import inspect
import typing
from functools import partial
from logging import getLogger
from typing import Any, Callable, Iterable

if typing.TYPE_CHECKING:
    from orion.client import ExperimentClient
    from orion.core.worker.experiment import Experiment
    from orion.core.worker.trial import Trial
    from orion.storage.base import BaseStorageProtocol


log = getLogger(__file__)

from ..experiment_config import ExperimentConfig


class KnowledgeBase:
    """Combination of a Storage containing previous experiments and an optional similarity metric.

    The Knowledge base is used to fetch previous experiments and their trials, which are then used
    to initialize (a.k.a. "warm-start" the hyper-parameter optimization algorithm.

    By default, the HPO algorithms of Orion are unable to reuse trials if they are not compatible
    with the space of the current (a.k.a. "target") experiment. Therefore, when warm-starting, a
    `MultiTaskWrapper` is added, which enables a limited form of warm-starting for all algorithms,
    by only considering the trials which are compatible with the target experiment, and training
    the algorithm in a multi-task fashion.

    When passed, the similarity function will be used to order the experiments such that the most
    related experiments are given first.
    """

    def __init__(
        self,
        storage: BaseStorageProtocol,
        similarity_metric: Callable[[ExperimentConfig, ExperimentConfig], float]
        | None = None,
    ):
        self.storage = storage
        self.similarity_metric = similarity_metric

    def get_related_trials(
        self,
        target_experiment: Experiment | ExperimentClient | ExperimentConfig,
        max_trials: int | None = None,
    ) -> list[tuple[ExperimentConfig, list[Trial]]]:
        """Retrieve the trials from experiments 'similar' to `target_experiment`.

        When `max_trials` is given, only up to `max_trials` are returned in total.

        Parameters
        ----------
        target_experiment : Union[Experiment, ExperimentClient, ExperimentConfig]
            The target experiment, or experiment client, or experiment configuration.
        max_trials : int, optional
            Maximum total number of trials to fetch. By default `None`, in which case
            all trials are returned.

        Returns
        -------
        list[tuple[ExperimentConfig, list[Trial]]]
            A list of tuples of experiments and the similar trials from that experiment.
        """
        if not isinstance(target_experiment, dict):
            target_experiment = target_experiment.configuration

        experiment_configs = self.storage.fetch_experiments({})
        if self.similarity_metric:
            # Order the experiments with decreasing similarity w.r.t the target experiment.
            sorting_function = partial(self.similarity_metric, target_experiment)
            experiment_configs.sort(key=sorting_function, reverse=True)

        return self._get_trials(experiment_configs, max_trials=max_trials)

    @property
    def n_stored_experiments(self) -> int:
        """Returns the current number of experiments registered the Knowledge base."""
        return len(self.storage.fetch_experiments({}))

    def _get_trials(
        self, experiment_configs: Iterable[ExperimentConfig], max_trials: int | None
    ) -> list[tuple[ExperimentConfig, list[Trial]]]:
        """Takes at most `max_trials` trials from the given experiments.

        If `max_trials` is None, returns all trials from all given experiments.
        """
        related_trials: list[tuple[ExperimentConfig, list[Trial]]] = []
        total_trials_so_far = 0
        for experiment_config in experiment_configs:
            experiment_id = experiment_config["_id"]
            # TODO: Is the experiment id always a string? or can it also be an integer?
            if experiment_id is None:
                continue
            trials = self.storage.fetch_trials(uid=str(experiment_id))

            if max_trials is not None:
                remaining = max_trials - total_trials_so_far
                if len(trials) >= remaining:
                    # Can only add some of the trials.
                    trials = trials[:remaining]

            related_trials.append((experiment_config, trials))
            total_trials_so_far += len(trials)
        return related_trials

    @property
    def configuration(self) -> dict[str, Any]:
        """Returns the configuration of the knowledge base.

        By default, returns a dictionary containing the attributes of `self` which are also
        constructor arguments.
        """
        # Note: This is a bit extra, but it will also work for subclasses of KnowledgeBase.
        init_signature = inspect.signature(type(self).__init__)
        init_arguments_attributes = {
            name: getattr(self, name)
            for name in init_signature.parameters
            if hasattr(self, name)
        }
        # Get the configuration of attributes if needed.
        init_argument_configurations = {
            name: value.configuration if hasattr(value, "configuration") else value
            for name, value in init_arguments_attributes.items()
        }
        return {type(self).__qualname__: init_argument_configurations}
