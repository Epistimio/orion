""" Defines the Quadratics task, as described in section 4.2 of the ABLR paper.

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, Optional, List

import numpy as np
from orion.benchmark.task.base import BaseTask

logger = get_logger(__name__)


@dataclass
class QuadraticsTaskHParams:
    """ Hyper-parameters of the Quadratics task. """

    x0: float
    x1: float
    x2: float


@dataclass
class QuadraticsTaskHparamsWithContext(QuadraticsTaskHParams):
    """ Hyper-parameters of the Quadratics task that also include the context vector.

    The context vector in this case are the coefficients of the quadratic.
    
    NOTE: This adds these entries to the space, but they are ignored when evaluating a given point
    (they aren't used by the `__call__` method of QuadraticsTask).
    """

    a2: float
    a1: float
    a0: float


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
    with_context : bool, optional
        Wether to append the values of (a2, a1, a0) to the points sampled from this task. 
    """

    def __init__(
        self,
        max_trials: int,
        a2: float = None,
        a1: float = None,
        a0: float = None,
        seed: int = None,
        with_context: bool = False,
    ):
        super().__init__(max_trials=max_trials)
        self.seed: Optional[int] = seed
        self.rng = np.random.default_rng(self.seed)
        # Note: rounding to 4 decimals just to prevent some potential issues
        # related to the string representation of the task later on.
        self.a2 = a2 if a2 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a1 = a1 if a1 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a0 = a0 if a0 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.with_context = with_context

    def get_search_space(self) -> Dict[str, str]:
        space = dict(
            x0="uniform(-100., 100., discrete=False)",
            x1="uniform(-100., 100., discrete=False)",
            x2="uniform(-100., 100., discrete=False)",
        )
        if self.with_context:
            space.update(
                {
                    "a2": "uniform(-10.0, 10.0, discrete=False)",
                    "a1": "uniform(-10.0, 10.0, discrete=False)",
                    "a0": "uniform(-10.0, 10.0, discrete=False)",
                }
            )
        return space

    def __repr__(self) -> str:
        return (
            f"QuadraticsTask(max_trials={self.max_trials}, a0={self.a0}, a1={self.a1}, "
            f"a2={self.a2}, seed={self.seed}, with_context={self.with_context})"
        )

    def call(self, x: np.ndarray) -> List[Dict]:
        if self.with_context and x.shape == (6,):
            # Discard the 'extra' values.
            x = x[..., :3]

        if x.shape != (3,):
            raise ValueError(
                f"Expected inputs to have shape (3,){' or (6,)' if self.with_context else ''}, but "
                f"got shape {x.shape} instead."
            )
        y = 0.5 * self.a2 * (x ** 2).sum() + self.a1 * x.sum() + self.a0
        # NOTE: We could also easily give back the gradient.
        return [dict(name="quadratics", type="objective", value=y)]
