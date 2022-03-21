# -*- coding: utf-8 -*-
"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
from __future__ import annotations

import copy
from logging import getLogger as get_logger
from typing import Any, Generic, Optional, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.registry import Registry, RegistryMapping
from orion.algo.space import Space
from orion.core.worker.transformer import TransformedSpace
from orion.core.worker.trial import Trial

logger = get_logger(__name__)

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)

# pylint: disable=too-many-public-methods
class SpaceTransformAlgoWrapper(Generic[AlgoType]):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    Parameters
    ----------
    algorithm: instance of `BaseAlgorithm`
        Algorithm to be wrapped.
    space : `orion.algo.space.Space`
       The original definition of a problem's parameters space.
    algorithm_config : dict
       Configuration for the algorithm.

    """

    def __init__(self, algorithm: AlgoType, space: Space):
        self._space = space
        self.algorithm: AlgoType = algorithm
        self.registry = Registry()
        self.registry_mapping = RegistryMapping(
            original_registry=self.registry,
            transformed_registry=self.algorithm.registry,
        )

    @property
    def original_space(self) -> Space:
        """The original space (before transformations).
        This is exposed to the outside, but not to the wrapped algorithm.
        """
        return self._space

    @property
    def transformed_space(self) -> TransformedSpace:
        """The transformed space (after transformations).
        This is only exposed to the wrapped algo, not to classes outside of this.
        """
        return self.algorithm.space

    def seed_rng(self, seed):
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        # TODO: There's currently some duplicates between:
        # - self.registry_mapping.original_registry and self.registry
        # - self.registry_mapping.transformed_registry and self.algorithm.registry
        return copy.deepcopy(
            {
                "algorithm": self.algorithm.state_dict,
                "registry": self.registry.state_dict,
                "registry_mapping": self.registry_mapping.state_dict,
            }
        )

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        state_dict = copy.deepcopy(state_dict)
        self.algorithm.set_state(state_dict["algorithm"])
        self.registry.set_state(state_dict["registry"])
        self.registry_mapping.set_state(state_dict["registry_mapping"])

    def suggest(self, num: int) -> list[Trial] | None:
        """Suggest a `num` of new sets of parameters.

        Parameters
        ----------
        num: int
            Number of points to suggest. The algorithm may return less than the number of points
            requested.

        Returns
        -------
        list of trials or None
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        transformed_trials = self.algorithm.suggest(num)

        if transformed_trials is None:
            return None

        trials = []
        for transformed_trial in transformed_trials:
            if transformed_trial not in self.transformed_space:
                raise ValueError(
                    f"Trial {transformed_trial.id} not contained in space:\n"
                    f"Params: {transformed_trial.params}\n"
                    f"Space: {self.transformed_space}"
                )
            original = self.transformed_space.reverse(transformed_trial)
            if original in self.registry:
                # We already have a trial that is equivalent to this one.
                # Copy over the status and results and observe it.
                transformed_trial = _copy_status_and_results(
                    original_trial=original, transformed_trial=transformed_trial
                )
                self.algorithm.observe([transformed_trial])
            else:
                # We haven't seen this trial before. Register it.
                trials.append(original)

            # Register the trial in our registry.
            self.registry.register(original)
            # Here we register the transformed trial just in case the algorithm hasn't already
            # registered the trial in its `suggest`.
            self.algorithm.register(transformed_trial)
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
            self.registry.register(trial)

            # Get the known transformed trials that correspond to this original.
            transformed_trials = self.registry_mapping.get_trials(trial)

            # Also transfer the status and results of `trial` to the equivalent transformed trials.
            transformed_trials = [
                _copy_status_and_results(
                    original_trial=trial, transformed_trial=transformed_trial
                )
                for transformed_trial in transformed_trials
            ]

            if not transformed_trials:
                # `trial` is a new, original trial that wasn't suggested by the algorithm. (This
                # might happen when an insertion is done according to @bouthilx)
                transformed_trial = self.transformed_space.transform(trial)
                transformed_trial = _copy_status_and_results(
                    original_trial=trial, transformed_trial=transformed_trial
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

    def has_suggested(self, trial: Trial) -> bool:
        """Whether the algorithm has suggested a given trial.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_suggested`
        """
        return self.registry.has_suggested(trial)

    def has_observed(self, trial: Trial) -> bool:
        """Whether the algorithm has observed a given trial.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_observed`
        """
        return self.registry.has_observed(trial)

    @property
    def n_suggested(self) -> int:
        """Number of trials suggested by the algorithm"""
        return len(self.registry)

    @property
    def n_observed(self) -> int:
        """Number of completed trials observed by the algorithm."""
        return sum(
            trial.status in ("completed", "broken") for trial in self.registry.values()
        )

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        n_suggested = self.n_suggested
        fidelity_index = self.fidelity_index
        if fidelity_index is not None:
            # When there is a fidelity dimension, we only count the trials with the max fidelity
            # value when comparing to the space cardinality, i.e., the algo is done if it has
            # suggested all points of the space, with the maximum fidelity value.
            n_suggested_with_max_fidelity = 0
            fidelity_dim = self.original_space[fidelity_index]
            _, max_fidelity_value = fidelity_dim.interval()
            for trial in self.registry.values():
                fidelity_value = trial.params[fidelity_index]
                if fidelity_value >= max_fidelity_value:
                    n_suggested_with_max_fidelity += 1
            n_suggested = n_suggested_with_max_fidelity
        if n_suggested >= self.original_space.cardinality:
            return True

        if hasattr(self, "max_trials"):
            max_trials: int | float = self.max_trials  # type: ignore
            if self.n_observed > max_trials:
                return True
        return self.algorithm.is_done

    def score(self, trial: Trial) -> float:
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        self._verify_trial(trial)
        return self.algorithm.score(self.transformed_space.transform(trial))

    def judge(self, trial: Trial, measurements: Any) -> dict | None:
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        self._verify_trial(trial)
        return self.algorithm.judge(
            self.transformed_space.transform(trial), measurements
        )

    def should_suspend(self, trial: Trial) -> bool:
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        self._verify_trial(trial)
        return self.algorithm.should_suspend(trial)

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        # TODO: Return a dict with the wrapped algo's configuration instead?
        return self.algorithm.configuration

    @property
    def space(self) -> Space:
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space

    def get_id(self, trial: Trial, ignore_fidelity: bool = False) -> str:
        """Compute a unique hash for a trial based on params"""
        # TODO: Double-check this with @bouthilx: Should we check the registry mapping? Or create a
        # new transformed trial?
        transformed_trial = self.transformed_space.transform(trial)
        return self.algorithm.get_id(transformed_trial, ignore_fidelity=ignore_fidelity)

    @property
    def fidelity_index(self) -> str | None:
        """Compute the index of the space where fidelity is.

        Returns None if there is no fidelity dimension.
        """
        return self.algorithm.fidelity_index

    def _verify_trial(self, trial: Trial, space: Optional[Space] = None) -> None:
        if space is None:
            space = self.space

        if trial not in space:
            raise ValueError(
                f"Trial {trial.id} not contained in space:"
                f"\nParams: {trial.params}\nSpace: {space}"
            )


def _copy_status_and_results(original_trial: Trial, transformed_trial: Trial) -> Trial:
    """Copies the results, status, and other data from `transformed_trial` to `original_trial`."""
    transformed_trial = copy.deepcopy(transformed_trial)
    transformed_trial.status = original_trial.status
    transformed_trial.end_time = original_trial.end_time
    if original_trial.results:
        transformed_trial.results = original_trial.results
    if original_trial.exp_working_dir:
        transformed_trial.exp_working_dir = original_trial.exp_working_dir
    return transformed_trial
