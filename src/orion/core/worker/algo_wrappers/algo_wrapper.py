""" ABC for a wrapper around an Algorithm. """
from __future__ import annotations

import copy
from logging import getLogger as get_logger
from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.trial import Trial

if TYPE_CHECKING:
    from orion.core.worker.experiment_config import ExperimentConfig

logger = get_logger(__name__)

AlgoT = TypeVar("AlgoT", bound=BaseAlgorithm)
WrappedT = TypeVar("WrappedT", bound=BaseAlgorithm)

# pylint: disable=too-many-public-methods
class AlgoWrapper(BaseAlgorithm, Generic[AlgoT]):
    """Base class for a Wrapper around an algorithm."""

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space)
        self._algorithm = algorithm

    def suggest(self, num: int) -> list[Trial]:
        trials = self.algorithm.suggest(num)
        for trial in trials:
            self.register(trial)
        return trials

    def observe(self, trials: list[Trial]) -> None:
        for trial in trials:
            self.register(trial)
        self.algorithm.observe(trials)

    @classmethod
    def transform_space(cls, space: Space) -> Space:
        """Transform an (outer) space, returning the (inner) space of the wrapped algorithm.

        The output of this classmethod will be used as `self.algorithm.space`, when we create an
        instance of `cls`.
        """
        return space

    @property
    def algorithm(self) -> AlgoT:
        """Returns the wrapped algorithm."""
        return self._algorithm

    @property
    def unwrapped(self):
        """Returns the unwrapped algorithm."""
        return self.algorithm.unwrapped

    def unwrap(self, target_type: type[WrappedT]) -> WrappedT:
        """Unwrap until the given type of wrapper or algorithm is encountered.

        If it isn't, raises a RuntimeError.
        """
        node = self
        while isinstance(node, AlgoWrapper):
            if isinstance(node, target_type):
                return node
            node = node.algorithm
        if isinstance(node, target_type):
            return node
        raise RuntimeError(
            f"Unable to find a wrapper or algorithm of type {target_type} in {self}"
        )

    @property
    def max_trials(self) -> int | None:
        """Maximum number of trials to run, or `None` when there is no limit."""
        return self._algorithm.max_trials

    @max_trials.setter
    def max_trials(self, value: int | None) -> None:
        """Sets the maximum number of trials to run on the algo wrapper.

        NOTE: This may or may not be propagated to the wrapped algorithm, depending on the type of
        wrapper.
        """
        self._algorithm.max_trials = value

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm.

        AlgoWrappers should overwrite this and add any additional state they own.
        """
        # Get the state of the wrapper's registry, via `BaseAlgorithm.state_dict`
        state_dict = super().state_dict
        # Add the state of the wrapped algorithm as a nested dictionary.
        state_dict["algorithm"] = copy.deepcopy(self.algorithm.state_dict)
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm

        AlgoWrappers should overwrite this and restore any additional state they own.
        """
        super().set_state(state_dict)
        self.algorithm.set_state(copy.deepcopy(state_dict["algorithm"]))

    def __repr__(self) -> str:
        algo_repr = repr(self.algorithm)
        if not (algo_repr.startswith("<") and algo_repr.endswith(">")):
            algo_repr = f"<{algo_repr}>"
        return f"{type(self).__qualname__}{algo_repr}"

    @property
    def is_done(self) -> bool:
        return super().is_done or self.algorithm.is_done

    @property
    def configuration(self) -> dict:
        # NOTE: The algo wrapper shouldn't have any configuration for now.
        # Here we just return the algo's config:
        return self.algorithm.configuration

    @property
    def fidelity_index(self) -> str | None:
        return self.algorithm.fidelity_index

    def should_suspend(self, trial: Trial) -> bool:
        return self.algorithm.should_suspend(trial)

    def warm_start(self, warm_start_trials: list[tuple[ExperimentConfig, list[Trial]]]):
        """Warm start the HPO algorithm by observing the given _related_ trials from other tasks."""
        from orion.core.worker.warm_start.warm_starteable import is_warmstarteable

        if not is_warmstarteable(self.algorithm):
            raise RuntimeError(
                f"The wrapped algorithm ({self.algorithm}) does not support warm starting with "
                f"trials from prior experiments."
            )
        self.algorithm.warm_start(warm_start_trials)

    def register(self, trial: Trial) -> None:
        super().register(trial)
