""" Defines the Quadratics task, as described in section 4.2 of the ABLR paper.

[1] [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)
"""
from logging import getLogger as get_logger
from typing import Dict, Optional, List

import numpy as np
from orion.benchmark.task.base import BenchmarkTask

logger = get_logger(__name__)


class QuadraticsTask(BenchmarkTask):
    """ Simple task consisting of a quadratic with three coefficients, as described in ABLR.

    Notes
    -----
    In the paper, section 4.2, the function is defined over R^3, but here the bounds are limited to
    [-100,100] for now since otherwise y can be enormous.

    Parameters
    ----------
    max_trials : int
        Maximum number of tr
    a_2 : float, optional
        a_2 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval,
        rounded to 4 decimal points.
    a_1 : float, optional
        a_1 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval,
        rounded to 4 decimal points.
    a_0 : float, optional
        a_0 coefficient, by default None, in which case it is sampled in the [0.1, 10.0] interval,
        rounded to 4 decimal points.
    seed : int, optional
        Random seed, by default None
    with_context : bool, optional
        Whether to append the values of (`a_2`, `a_1`, and `a_0`) to the points that are sampled
        from this task. When set to `True`, any value of `a_2`, `a_1`, or `a_0` passed to `call` is
        ignored.
    """

    def __init__(
        self,
        max_trials: int,
        a_2: float = None,
        a_1: float = None,
        a_0: float = None,
        seed: int = None,
        with_context: bool = False,
    ):
        super().__init__(max_trials=max_trials)
        self.seed: Optional[int] = seed
        self.rng = np.random.default_rng(self.seed)
        # Note: rounding to 4 decimals just to prevent some potential issues
        # related to the string representation of the task later on.
        self.a_2 = a_2 if a_2 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a_1 = a_1 if a_1 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.a_0 = a_0 if a_0 is not None else round(self.rng.uniform(0.1, 10.0), 4)
        self.with_context = with_context

    def get_search_space(self) -> Dict[str, str]:
        space = dict(
            x_2="uniform(-100., 100., precision=4, discrete=False)",
            x_1="uniform(-100., 100., precision=4, discrete=False)",
            x_0="uniform(-100., 100., precision=4, discrete=False)",
        )
        if self.with_context:
            space.update(
                {
                    "a_2": "uniform(-10.0, 10.0, precision=4, discrete=False)",
                    "a_1": "uniform(-10.0, 10.0, precision=4, discrete=False)",
                    "a_0": "uniform(-10.0, 10.0, precision=4, discrete=False)",
                }
            )
        return space

    def __repr__(self) -> str:
        return (
            f"QuadraticsTask(max_trials={self.max_trials}, a_0={self.a_0}, a_1={self.a_1}, "
            f"a_2={self.a_2}, seed={self.seed}, with_context={self.with_context})"
        )

    def call(
        self,
        x_0: float,
        x_1: float,
        x_2: float,
        a_2: float = None,
        a_1: float = None,
        a_0: float = None,
    ) -> List[Dict]:
        x = np.array([x_0, x_1, x_2])
        y = 0.5 * self.a_2 * (x ** 2).sum() + self.a_1 * x.sum() + self.a_0
        return [dict(name="quadratics", type="objective", value=y)]
