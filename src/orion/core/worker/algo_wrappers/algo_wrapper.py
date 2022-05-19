""" ABC for a wrapper around an Algorithm. """
from __future__ import annotations

import copy
from contextlib import contextmanager
from logging import getLogger as get_logger
from typing import Any, Generic, Sequence, TypeVar

from typing_extensions import ParamSpec

from orion.algo.base import BaseAlgorithm
from orion.algo.registry import RegistryMapping
from orion.algo.space import Space
from orion.core.worker.trial import Trial

logger = get_logger(__name__)

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)
P = ParamSpec("P")


class AlgoWrapper(BaseAlgorithm, Generic[AlgoType]):
    """Base class for a Wrapper around an algorithm.

    Applies some transformation to the incoming trials before they get passed to the underlying
    algorithm, and applies the reverse transformation to the trials suggested by the algorithm
    before they are suggested by the wrapper.
    """

    def __init__(self, space: Space, algorithm: AlgoType):
        # NOTE: This field is created automatically in the BaseAlgorithm class.
        super().__init__(space)
        self._algorithm = algorithm
        assert not isinstance(algorithm, dict), algorithm
        self.registry_mapping = RegistryMapping(
            original_registry=self.registry,
            transformed_registry=self.algorithm.registry,
        )

    def transform(self, trial: Trial) -> Trial:
        """Transform a trial from the space of the wrapper to the space of the wrapped algorithm."""
        return trial

    def reverse_transform(self, trial: Trial) -> Trial:
        """Transform a trial from the space of the wrapped algo to the space of the wrapper."""
        return trial

    @classmethod
    def transform_space(cls, space: Space) -> Space:
        """Transform the space, so that the algorithm that is passed to the constructor already
        has the right space.
        """
        return space

    @property
    def algorithm(self) -> AlgoType:
        """Returns the wrapped algorithm.

        Returns
        -------
        AlgoType
            The wrapped algorithm.
        """
        return self._algorithm

    @property
    def unwrapped(self):
        """Returns the unwrapped algorithm (the root).

        Returns
        -------
        BaseAlgorithm
            The unwrapped `BaseAlgorithm` instance.
        """
        return self.algorithm.unwrapped

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm.

        AlgoWrappers should overwrite this and add any additional state they are responsible for.
        """
        state_dict = super().state_dict
        state_dict.update(
            {
                "algorithm": copy.deepcopy(self.algorithm.state_dict),
                "registry_mapping": self.registry_mapping.state_dict,
            }
        )
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        AlgoWrappers should overwrite this and restore any additional state they have.
        """
        state_dict = copy.deepcopy(state_dict)
        super().set_state(state_dict)
        self.algorithm.set_state(state_dict["algorithm"])
        self.registry_mapping.set_state(state_dict["registry_mapping"])

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
                    f"Trial {transformed_trial.id} not contained in transformed space:\n"
                    f"Params: {transformed_trial.params}\n"
                    f"Space: {self.algorithm.space}"
                )
            original = self.reverse_transform(transformed_trial)

            if self.has_suggested(original):
                logger.debug(
                    "Already suggested or observed a trial that matches %s in the registry.",
                    original,
                )
                # We already have a trial that is equivalent to this one.
                # Fetch the actual trial (with the status and possibly results)
                original = self.registry.get_existing(original)
                logger.debug("Matching trial (with results/status): %s", original)

                # Copy over the status and any results from the original to the transformed trial
                # and observe it.
                transformed_trial = _copy_status_and_results(
                    trial_with_status=original, trial_with_params=transformed_trial
                )
                logger.debug(
                    "Transformed trial (with status/results): %s", transformed_trial
                )
                self.algorithm.observe([transformed_trial])

            else:
                logger.debug(
                    "New suggestion: %s",
                    original,
                )
                # We haven't seen this trial before. Register it.
                self.registry.register(original)
                trials.append(original)

            # NOTE: Here we DON'T register the transformed trial, we let the algorithm do it
            # itself in its `suggest`.
            # Register the equivalence between these trials.
            self.registry_mapping.register(original, transformed_trial)

        return trials

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__}{self.algorithm}>"

    def observe(self, trials: list[Trial]) -> None:
        """Observe evaluated trials.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        # For each trial in the original space, find the suggestions from the algo that match it.
        # Then, we make the wrapped algo observe each equivalent trial, with the updated status.
        for trial in trials:
            # Update the status of this trial in the registry (if it was suggested), otherwise set
            # it in the registry (for example in testing when we force the algo to observe a trial).
            self.registry.register(trial)

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
                self.registry.register(trial)
                self.registry_mapping.register(trial, transformed_trial)

            self.algorithm.observe(transformed_trials)

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm"""
        # TODO: Double check that this makes sense. We only count the number of trials form the
        # target task that we suggested.
        return super().n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm."""
        return super().n_observed

    @property
    def is_done(self) -> bool:
        """Return True, if an algorithm holds that there can be no further improvement."""
        return super().is_done or self.algorithm.is_done

    def score(self, trial: Trial) -> float:
        return self.algorithm.score(self.transform(trial))

    def judge(self, trial: Trial, measurements: Any) -> dict | None:
        return self.algorithm.judge(self.transform(trial), measurements)

    def should_suspend(self, trial: Trial) -> bool:
        return self.algorithm.should_suspend(self.transform(trial))

    @property
    def configuration(self) -> dict:
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        Subclasses should overwrite this method and add any of their state.
        """
        # TODO: Do we also save the algorithm wrapper's configuration here? Or only the algo's?
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith("_"):  # Do not log _space or others in conf
                continue
            value = getattr(self, attrname)
            if attrname == "algorithm":
                # Store the configuration of the algorithm instead of the algorithm itself.
                value = value.configuration
            dict_form[attrname] = value

        return {self.__class__.__name__.lower(): dict_form}

    @contextmanager
    def warm_start_mode(self):
        """Context manager that is used while using points from similar experiments to
        bootstrap (warm-start) the algorithm.

        The idea behind this is that we don't want the algorithm to modify its state the
        same way it would if it were observing regular points. For example, the number
        of "used" trials shouldn't increase, etc.

        New Algorithms or Algo wrappers can implement this method to control how the
        state of the algo is affected by observing trials from other tasks than the
        current (target) task.
        """
        with self.algorithm.warm_start_mode():
            yield


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
