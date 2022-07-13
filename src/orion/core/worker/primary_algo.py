"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
from __future__ import annotations

import copy
from logging import getLogger as get_logger
from typing import Any, Generic, Sequence, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.registry import Registry, RegistryMapping
from orion.algo.space import Space
from orion.core.worker.transformer import TransformedSpace
from orion.core.worker.trial import Trial

logger = get_logger(__name__)

AlgoT = TypeVar("AlgoT", bound=BaseAlgorithm)


def create_algo(
    algo_type: type[AlgoT],
    space: Space,
    **algo_kwargs,
) -> SpaceTransformAlgoWrapper[AlgoT]:
    """Creates an algorithm of the given type, taking care of transforming the space if needed."""
    original_space = space
    from orion.core.worker.transformer import build_required_space

    # TODO: We could perhaps eventually *not* wrap the algorithm if it doesn't require any
    # transformations. For now we just always wrap it.
    transformed_space = build_required_space(
        space,
        type_requirement=algo_type.requires_type,
        shape_requirement=algo_type.requires_shape,
        dist_requirement=algo_type.requires_dist,
    )
    algorithm = algo_type(transformed_space, **algo_kwargs)
    wrapped_algo = SpaceTransformAlgoWrapper(algorithm=algorithm, space=original_space)
    return wrapped_algo


# pylint: disable=too-many-public-methods
class SpaceTransformAlgoWrapper(BaseAlgorithm, Generic[AlgoT]):
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

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space=space)
        self.algorithm: AlgoT = algorithm
        self.registry = Registry()
        self.registry_mapping = RegistryMapping(
            original_registry=self.registry,
            transformed_registry=self.algorithm.registry,
        )
        self.max_suggest_attempts = 100

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

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
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

        for suggest_attempt in range(1, self.max_suggest_attempts + 1):
            transformed_trials: list[Trial] | None = self.algorithm.suggest(num)
            transformed_trials = transformed_trials or []

            for transformed_trial in transformed_trials:
                if transformed_trial not in self.transformed_space:
                    raise ValueError(
                        f"Trial {transformed_trial.id} not contained in space:\n"
                        f"Params: {transformed_trial.params}\n"
                        f"Space: {self.transformed_space}"
                    )
                original = self.transformed_space.reverse(transformed_trial)
                if transformed_trial.parent:

                    original_parent = get_original_parent(
                        self.algorithm.registry,
                        self.transformed_space,
                        transformed_trial.parent,
                    )
                    if original_parent.id not in self.registry:
                        raise KeyError(
                            f"Parent with id {original_parent.id} is not registered."
                        )

                    original.parent = original_parent.id
                if original in self.registry:
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
                        original_trial=original, transformed_trial=transformed_trial
                    )
                    logger.debug(
                        "Transformed trial (with results/status): %s", transformed_trial
                    )
                    self.algorithm.observe([transformed_trial])
                else:
                    # We haven't seen this trial before. Register it.
                    self.registry.register(original)
                    trials.append(original)

                # NOTE: Here we DON'T register the transformed trial, we let the algorithm do it
                # itself in its `suggest`.
                # Register the equivalence between these trials.
                self.registry_mapping.register(original, transformed_trial)

            if trials:
                if suggest_attempt > 1:
                    logger.debug(
                        f"Succeeded in suggesting new trials after {suggest_attempt} attempts."
                    )
                return trials

            if self.is_done:
                logger.debug(
                    f"Algorithm is done! (after {suggest_attempt} sampling attempts)."
                )
                break

        logger.warning(
            f"Unable to sample a new trial from the algorithm, even after "
            f"{self.max_suggest_attempts} attempts! Returning an empty list."
        )
        return []

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
        return sum(self.has_observed(trial) for trial in self.registry)

    @property
    def is_done(self) -> bool:
        """Return True if the wrapper or the wrapped algorithm is done."""
        return super().is_done or self.algorithm.is_done

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
    def configuration(self) -> dict:
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        # TODO: Return a dict with the wrapped algo's configuration instead?
        # return {
        #     type(self).__qualname__: {
        #         "space": self.space.configuration,
        #         "algorithm": {self.algorithm.configuration},
        #     }
        # }
        return self.algorithm.configuration

    @property
    def space(self) -> Space:
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space

    @property
    def fidelity_index(self) -> str | None:
        """Compute the index of the space where fidelity is.

        Returns None if there is no fidelity dimension.
        """
        return self.algorithm.fidelity_index

    def _verify_trial(self, trial: Trial, space: Space | None = None) -> None:
        if space is None:
            space = self.space

        if trial not in space:
            raise ValueError(
                f"Trial {trial.id} not contained in space:"
                f"\nParams: {trial.params}\nSpace: {space}"
            )


def get_original_parent(
    registry: Registry, transformed_space: TransformedSpace, trial_parent_id: str
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


def _copy_status_and_results(original_trial: Trial, transformed_trial: Trial) -> Trial:
    """Copies the results, status, and other data from `transformed_trial` to `original_trial`."""
    new_transformed_trial = copy.deepcopy(original_trial)
    # pylint: disable=protected-access
    new_transformed_trial._params = copy.deepcopy(transformed_trial._params)
    new_transformed_trial.experiment = None
    new_transformed_trial.parent = transformed_trial.parent
    return new_transformed_trial
