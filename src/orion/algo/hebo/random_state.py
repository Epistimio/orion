""" Utility class for saving / restoring state of the global random number generators. """
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError as err:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True


@dataclass(frozen=True)
class RandomState:
    """Immutable dataclass that holds the state of the various random number generators."""

    random_state: Any
    """ RNG state for the `random` module. """

    numpy_rng_state: Dict[str, Any]
    """ RNG state for the `numpy.random` module. """

    if TORCH_INSTALLED:
        torch_rng_state: torch.Tensor
        """ RNG state for the `torch` module. """

        torch_cuda_rng_state: List[torch.Tensor]
        """ RNG states for the `torch.cuda` module, for each cuda device available. """

    base_seed: Optional[int] = None
    """Base seed that was used to create this object (the `seed` argument to `RandomState.seed`)."""

    @classmethod
    def current(cls) -> RandomState:
        """Returns the current random state.

        NOTE: The `base_seed` of the returned object will be None.
        """
        kwargs: dict[str, Any] = dict(
            random_state=random.getstate(),
            numpy_rng_state=np.random.get_state(),
        )
        if TORCH_INSTALLED:
            kwargs.update(
                torch_rng_state=torch.get_rng_state(),
                torch_cuda_rng_state=torch.cuda.get_rng_state_all(),
            )
        return cls(**kwargs)

    def set(self) -> None:
        """Sets the random state using the values in `self`."""
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_rng_state)
        if TORCH_INSTALLED:
            torch.random.set_rng_state(self.torch_rng_state)
            torch.cuda.set_rng_state_all(self.torch_cuda_rng_state)

    @classmethod
    def seed(cls, base_seed: Optional[int]) -> RandomState:
        """Seeds all the RNGs using the given seed, and returns the resulting RandomState object."""
        random.seed(base_seed)
        numpy_seed = random.randint(0, 2**32 - 1)
        # NOTE: Always create the torch seed, even if torch is not installed.
        torch_seed = random.randint(0, 2**32 - 1)
        np.random.seed(numpy_seed)
        if TORCH_INSTALLED:
            torch.random.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)

        random_state = cls.current()
        if base_seed is not None:
            # Add the base_seed property in this case.
            random_state = replace(random_state, base_seed=base_seed)
        return random_state
