""" Defines the Quadratics task, as described in section 4.2 of the ABLR paper.

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, Optional

import numpy as np
from orion.benchmark.task.base import BaseTask

logger = get_logger(__name__)


@dataclass
class QuadraticsTaskHParams:
    # NOTE: In the paper, section 4.2, the function is defined over R^3, but I'm
    # limiting the bounds to [-100,100] for now since otherwise y can be enormous.
    x0: float
    x1: float
    x2: float


class QuadraticsTask(BaseTask):
    def __init__(
        self,
        max_trials: int,
        a2: float = None,
        a1: float = None,
        a0: float = None,
        seed: int = None,
    ):
        super().__init__(max_trials=max_trials)
        self.seed: Optional[int] = seed
        self.rng = np.random.default_rng(self.seed)
        # Note: rounding to 4 decimals just to prevent some potential issues
        # related to the string representation of the task later on.
        self.a2 = a2 if a2 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a1 = a1 if a1 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a0 = a0 if a0 is not None else round(self.rng.uniform(0.1, 10.0), 4)

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            x0="uniform(-100., 100., discrete=False)",
            x1="uniform(-100., 100., discrete=False)",
            x2="uniform(-100., 100., discrete=False)",
        )

    def with_context(self) -> "QuadraticsTaskWithContext":
        return QuadraticsTaskWithContext(
            max_trials=self.max_trials,
            a2=self.a2,
            a1=self.a1,
            a0=self.a0,
            seed=self.seed,
        )

    def __repr__(self) -> str:
        return f"QuadraticsTask(max_trials={self.max_trials}, a0={self.a0}, a1={self.a1}, a2={self.a2}, seed={self.seed})"

    def call(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] > 3:
            # Discard any 'extra' values.
            # NOTE: This is only really used in the 'with context' version of
            # this task below.
            x = x[..., :3]

        assert x.shape == (3,)
        y = 0.5 * self.a2 * (x ** 2).sum() + self.a1 * x.sum() + self.a0
        return y

    def gradient(self, x: np.ndarray) -> float:
        # NOTE: Unused, but we could also give back the gradient.
        assert x.shape[-1] == 3
        return self.a2 * x.sum(-1) + self.a1


@dataclass
class QuadraticsTaskWithContextHparams(QuadraticsTaskHParams):
    # This adds these entries to the space, but they aren't used in the
    # quadratic equation above. (`__call__` of QuadraticsTask.)
    a2: float
    a1: float
    a0: float


class QuadraticsTaskWithContext(QuadraticsTask):
    """ Same as the QuadraticsTask, but the samples also have the "context" 
    information added to them (a2, a1, a0).
    
    This is used to help demonstrate the effectiveness of multi-task models,
    since the observations are directly related to this context vector (since
    they are the coefficients of the quadratic) and makes the problem super easy
    to solve.
    """

    def __init__(
        self,
        max_trials: int,
        a2: float = None,
        a1: float = None,
        a0: float = None,
        seed: Optional[int] = None,
    ):
        super().__init__(max_trials=max_trials, a2=a2, a1=a1, a0=a0, seed=seed)

    @property
    def context_vector(self) -> np.ndarray:
        return np.asfarray([self.a2, self.a1, self.a0])

    def call(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] > 3:
            # Discard any 'extra' values.
            # NOTE: This is only really used in the 'with context' version of
            # this task below.
            x = x[..., :3]
        return super().call(x=x)

    def get_search_space(self) -> Dict[str, str]:
        space = super().get_search_space()
        # TODO: Not sure this is the best way of doing this: We want the algorithm to be trained
        # using points that include these dimensions, but we don't want it to try to learn them.
        space.update(
            {
                "a2": "uniform(-10.0, 10.0, discrete=False)",
                "a1": "uniform(-10.0, 10.0, discrete=False)",
                "a0": "uniform(-10.0, 10.0, discrete=False)",
            }
        )
        return space
