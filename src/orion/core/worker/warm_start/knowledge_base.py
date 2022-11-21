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
    from orion.core.worker.experiment_config import ExperimentConfig
    from orion.core.worker.trial import Trial
    from orion.storage.base import BaseStorageProtocol


log = getLogger(__file__)


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
        target_config: ExperimentConfig
        if isinstance(target_experiment, dict):
            target_config = target_experiment
        else:
            target_config = target_experiment.configuration

        experiment_configs = self.storage.fetch_experiments({})
        # NOTE: Here we remove the target experiment from the list of experiments.
        # This occurs because we use a singleton for the storage, so creating the target experiment
        # implicitly adds it to the storage (and to the knowledge base).
        experiment_configs = _remove_target_config(target_config, experiment_configs)

        if self.similarity_metric:
            # Order the experiments with decreasing similarity w.r.t the target experiment.
            sort_fn = partial(self.similarity_metric, target_config)
            experiment_configs = sorted(experiment_configs, key=sort_fn, reverse=True)

        return self._get_trials(experiment_configs, max_trials=max_trials)

    @property
    def n_stored_experiments(self) -> int:
        """Returns the current number of experiments registered in the Knowledge base.

        NOTE: IF the target experiment is using the same storage object as the KB, then this
        will also count the target experiment.
        """
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
            # NOTE: experiments were just fetched from storage, so they *always* have an ID and
            # fetch_trials will never return None.
            assert experiment_id is not None

            trials = self.storage.fetch_trials(uid=experiment_id)
            assert trials is not None

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
        # Note: This is perhaps a bit too generic. But it works for any class or subclass.
        # We could probably move this to a `Configured` mixin class eventually.
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


def _remove_target_config(
    target_exp_config: ExperimentConfig, exp_configs: list[ExperimentConfig]
) -> list[ExperimentConfig]:
    """Removes the config matching the target experiment from `exp_configs`, if present.

    NOTE: We unfortunately can't just use `exp_configs.remove(target_exp_config)` because the dicts
    might not match exactly, due to small mutations that happen when going in and out of Storage.
    For instance, the `working_dir` property changes type "" -> None, and other fields might also.

    Therefore we do a simple check based on the name, id, and version number of the experiment to
    determine if it is the same experiment as the target.
    """
    result: list[ExperimentConfig] = []
    for exp_config in exp_configs:
        if not (
            exp_config["_id"] == target_exp_config["_id"]
            and exp_config["name"] == target_exp_config["name"]
            and exp_config["version"] == target_exp_config["version"]
        ):
            result.append(exp_config)
    return result
