""" ABC for a wrapper around an Algorithm. """
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generic, Sequence, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.registry import RegistryMapping
from orion.algo.space import Space
from orion.core.worker.trial import Trial

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)


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
    def unwrapped(self) -> BaseAlgorithm:
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
        return self.algorithm.state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        AlgoWrappers should overwrite this and restore any additional state they have.
        """
        self.algorithm.set_state(state_dict)

    def suggest(self, num: int = 1) -> list[Trial]:
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        transformed_trials = self.algorithm.suggest(num)
        original_trials = []
        for transformed_trial in transformed_trials:
            original_trial = self.reverse_transform(transformed_trial)
            self.register(original_trial)
            original_trials.append(original_trial)
            # NOTE: Also register the trial equivalence in the registry mapping.
            self.registry_mapping.register(original_trial, transformed_trial)

        return original_trials

    def observe(self, trials: list[Trial]):
        """Observe the `results` of the evaluation of the `trials` in the
        process defined in user's script.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        transformed_trials: list[Trial] = []
        for trial in trials:
            self.register(trial)
            transformed_trial = self.transform(trial)
            transformed_trials.append(transformed_trial)
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
