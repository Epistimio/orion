""" ABC for a wrapper around an Algorithm. """
from __future__ import annotations

import copy
from contextlib import contextmanager
from logging import getLogger as get_logger
from typing import Generic, Sequence, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.trial import Trial

logger = get_logger(__name__)

AlgoT = TypeVar("AlgoT", bound=BaseAlgorithm)


# pylint: disable=too-many-public-methods
class AlgoWrapper(BaseAlgorithm, Generic[AlgoT]):
    """Base class for a Wrapper around an algorithm."""

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space)
        self._algorithm = algorithm

    def suggest(self, num: int) -> list[Trial]:
        return self.algorithm.suggest(num)

    def observe(self, trials: list[Trial]) -> None:
        return self.algorithm.observe(trials)

    @classmethod
    def transform_space(cls, space: Space) -> Space:
        """Transform the given space, before it is used to instantiate the wrapped algorithm."""
        return space

    @property
    def algorithm(self) -> AlgoT:
        """Returns the wrapped algorithm."""
        return self._algorithm

    @property
    def unwrapped(self):
        """Returns the unwrapped algorithm."""
        return self.algorithm.unwrapped

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
        return f"{type(self).__qualname__}<{repr(self.algorithm)}>"

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

    @contextmanager
    def warm_start_mode(self):
        """Context manager that is used while observing trials from similar experiments to
        bootstrap (warm-start) the algorithm.

        The idea behind this is that we might not want the algorithm to modify its state the
        same way it would if it were observing regular trials. For example, the number
        of "used" trials shouldn't increase, etc.

        New algorithms or algo wrappers can implement this method to control how the
        state of the algo is affected by observing trials from other tasks than the
        current (target) task.
        """
        with self.algorithm.warm_start_mode():
            yield
