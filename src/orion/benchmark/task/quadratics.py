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
    """ Hyper-parameters of the Quadratics task. """
    x0: float
    x1: float
    x2: float


class QuadraticsTask(BaseTask):
    """ Simple task consisting of a quadratic with three coefficients, as described in ABLR.
    
    NOTE: In the paper, section 4.2, the function is defined over R^3, but here
    the bounds are limited to [-100,100] for now since otherwise y can be enormous.

    Parameters
    ----------
    max_trials : int
        Maximum number of tr
    a2 : float, optional
        a2 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval.
    a1 : float, optional
        a1 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval.
    a0 : float, optional
        a0 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval.
    seed : int, optional
        Random seed, by default None
    """
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
        """Returns a quadratics task that also includes the task coefficients in its trials.

        Returns
        -------
        QuadraticsTaskWithContext
            A task with the same coefficients but whose space also contains the quadratic
            coefficients.
        """
        return QuadraticsTaskWithContext(
            max_trials=self.max_trials, a2=self.a2, a1=self.a1, a0=self.a0, seed=self.seed,
        )

    def __repr__(self) -> str:
        return f"QuadraticsTask(max_trials={self.max_trials}, a0={self.a0}, a1={self.a1}, a2={self.a2}, seed={self.seed})"

    def call(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (3,):
            raise ValueError(
                f"Expected inputs to have shape (3,), but got shape {x.shape} instead."
            )
        y = 0.5 * self.a2 * (x ** 2).sum() + self.a1 * x.sum() + self.a0
        # NOTE: We could also easily give back the gradient.
        return y


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
        if x.shape[-1] != 6:
            raise ValueError(
                f"Expected inputs to have shape (6,), but got shape {x.shape} instead."
            )
        # Discard the 'extra' values.
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
