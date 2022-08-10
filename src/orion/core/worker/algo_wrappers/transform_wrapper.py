""" Wrapper that applies a transformation to the trials suggested/observed by the algorithm."""
from __future__ import annotations

import copy
import typing
from abc import ABC, abstractmethod
from logging import getLogger as get_logger
from typing import Any

from orion.algo.registry import Registry, RegistryMapping
from orion.algo.space import Space
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoT, AlgoWrapper
from orion.core.worker.transformer import ReshapedSpace, TransformedSpace
from orion.core.worker.trial import Trial

if typing.TYPE_CHECKING:
    from orion.core.worker.experiment_config import ExperimentConfig

logger = get_logger(__name__)


class TransformWrapper(AlgoWrapper[AlgoT], ABC):
    """Wrapper around an algorithm that applies some transformation to the trials it suggests.

    The inverse transformation is applied to the observed trials before they get passed down to the
    wrapped algorithm's `observe` method.
    """

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space, algorithm)
        self.registry_mapping = RegistryMapping(
            original_registry=self.registry,
            transformed_registry=self.algorithm.registry,
        )

    @abstractmethod
    def transform(self, trial: Trial) -> Trial:
        """Transform a trial from the space of the wrapper to the space of the wrapped algorithm."""
        return trial

    @abstractmethod
    def reverse_transform(self, trial: Trial) -> Trial:
        """Transform a trial from the space of the wrapped algo to the space of the wrapper."""
        return trial

    def suggest(self, num: int) -> list[Trial]:
        """Suggest a `num` of new sets of parameters.

        Parameters
        ----------
        num: int
            Number of trials to suggest. The algorithm may return less than the number of trials
            requested.

        Returns
        -------
        list of trials
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return an empty list.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """

        trials: list[Trial] = []

        transformed_trials: list[Trial] = self.algorithm.suggest(num) or []

        for transformed_trial in transformed_trials:
            if transformed_trial not in self.algorithm.space:
                raise ValueError(
                    f"Trial {transformed_trial.id} not contained in space:\n"
                    f"Params: {transformed_trial.params}\n"
                    f"Space: {self.algorithm.space}"
                )
            original = self.reverse_transform(transformed_trial)

            if transformed_trial.parent:
                # NOTE: This block of code was previously in what has become the SpaceTransform
                # wrapper, which assumes the following:
                assert isinstance(
                    self.algorithm.space, (TransformedSpace, ReshapedSpace)
                )
                original_parent = get_original_parent(
                    registry=self.algorithm.registry,
                    transformed_space=self.algorithm.space,
                    trial_parent_id=transformed_trial.parent,
                )
                if original_parent.id not in self.registry:
                    raise KeyError(
                        f"Parent with id {original_parent.id} is not registered."
                    )

                original.parent = original_parent.id

            if self.has_suggested(original):
                logger.debug(
                    "Already have a trial that matches %s in the registry.",
                    original,
                )
                # We already have a trial that is equivalent to this one.
                # Fetch the actual trial (with the status and possibly results)
                original = self.registry.get_existing(original)
                logger.debug("Matching trial (with results/status): %s", original)

                # Copy over the status and results from the original to the transformed trial
                # and observe it.
                transformed_trial = _copy_status_and_results(
                    trial_with_status=original, trial_with_params=transformed_trial
                )
                logger.debug(
                    "Transformed trial (with results/status): %s", transformed_trial
                )
                self.algorithm.observe([transformed_trial])
            else:
                # We haven't seen this trial before. Register it.
                self.register(original)
                trials.append(original)

            # NOTE: Here we DON'T register the transformed trial, we let the algorithm do it
            # itself in its `suggest`.
            # Register the equivalence between these trials.
            self.registry_mapping.register(original, transformed_trial)
        return trials

    def observe(self, trials: list[Trial]) -> None:
        """Observe evaluated trials.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        # For each trial in the original space, find the suggestions from the algo that match it.
        # Then, we make the wrapped algo observe each equivalent trial, with the updated status.
        for trial in trials:
            # Update the status of this trial in the registry (if it was suggested), otherwise set
            # it in the registry (for example in testing when we force the algo to observe a trial).
            self.register(trial)

            # Get the known transformed trials that correspond to this original.
            transformed_trials = self.registry_mapping.get_trials(trial)

            # Also transfer the status and results of `trial` to the equivalent transformed trials.
            transformed_trials = [
                _copy_status_and_results(
                    trial_with_status=trial, trial_with_params=transformed_trial
                )
                for transformed_trial in transformed_trials
            ]

            if not transformed_trials:
                # `trial` is a new, original trial that wasn't suggested by the algorithm. (This
                # might happen when an insertion is done according to @bouthilx)
                transformed_trial = self.transform(trial)
                transformed_trial = _copy_status_and_results(
                    trial_with_status=trial, trial_with_params=transformed_trial
                )
                transformed_trials = [transformed_trial]
                logger.debug(
                    f"Observing trial {trial} (transformed as {transformed_trial}), even "
                    f"though it wasn't suggested by the algorithm."
                )
                # NOTE: @lebrice Here we don't want to store the transformed trial in the
                # algo's registry (by either calling `self.algorithm.register(transformed_trial)` or
                # `self.algorithm.registry.register(transformed_trial))`, because some algos can't
                # observe trials that they haven't suggested. We'd also need to perform all the
                # logic that the algo did in `suggest` (e.g. store it in a bracket for HyperBand).
                # Therefore we only register it in the wrapper, and store the equivalence between
                # these two trials in the registry mapping.
                self.register(trial)
                self.registry_mapping.register(trial, transformed_trial)

            self.algorithm.observe(transformed_trials)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm.

        AlgoWrappers should overwrite this and add any additional state they are responsible for.
        """
        state_dict = super().state_dict
        state_dict["registry_mapping"] = self.registry_mapping.state_dict
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        AlgoWrappers should overwrite this and restore any additional state they have.
        """
        super().set_state(state_dict)
        self.registry_mapping.set_state(state_dict["registry_mapping"])

    def score(self, trial: Trial) -> float:
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        return self.algorithm.score(self.transform(trial))

    def judge(self, trial: Trial, measurements: Any) -> dict | None:
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        return self.algorithm.judge(self.transform(trial), measurements)

    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentConfig, list[Trial]]]
    ) -> None:
        super().warm_start(
            [
                (experiment_info, [self.transform(trial) for trial in trials])
                for experiment_info, trials in warm_start_trials
            ]
        )


def get_original_parent(
    registry: Registry,
    transformed_space: TransformedSpace | ReshapedSpace,
    trial_parent_id: str,
) -> Trial:
    """Get the parent trial in original space based on parent id in transformed_space.

    If the parent trial also has a parent, then this function is called recursively
    to set the proper parent id in original space rather than transformed space.
    """
    try:
        parent = registry[trial_parent_id]
    except KeyError as e:
        raise KeyError(f"Parent with id {trial_parent_id} is not registered.") from e

    original_parent = transformed_space.reverse(parent)
    if original_parent.parent is None:
        return original_parent

    original_grand_parent = get_original_parent(
        registry, transformed_space, original_parent.parent
    )
    original_parent.parent = original_grand_parent.id
    return original_parent


def _copy_status_and_results(
    trial_with_status: Trial, trial_with_params: Trial
) -> Trial:
    """Copies the results, status, and other data from `source_trial` to `original_trial`.
    Returns a new Trial.
    """
    new_transformed_trial = copy.deepcopy(trial_with_status)
    # pylint: disable=protected-access
    new_transformed_trial._params = copy.deepcopy(trial_with_params._params)
    return new_transformed_trial
