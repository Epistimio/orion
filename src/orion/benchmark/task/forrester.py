"""Forrester Task from the Profet paper.

This Forrester class is based on a synthetic function, whereas the ForresterTask is baseed on a
meta-model trained on multiple such functions.

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate
benchmarking for hyperparameter optimization." Advances in Neural Information Processing Systems 32
(2019): 6270-6280.
"""
from typing import Dict, List

import numpy as np

from orion.benchmark.task.base import BenchmarkTask


class Forrester(BenchmarkTask):
    """Task based on the Forrester function, as described in https://arxiv.org/abs/1905.12982

    .. math:: f(x) = ((\alpha x - 2)^2) sin(\beta x - 4)


    Parameters
    ----------
    max_trials : int
        Maximum number of trials for this task.
    alpha : float, optional
        Alpha parameter used in the above equation, by default 0.5
    beta : float, optional
        Beta parameter used in the above equation, by default 0.5
    """

    def __init__(self, max_trials: int, alpha: float = 0.5, beta: float = 0.5):
        super().__init__(max_trials, alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta

    def call(self, x: float) -> List[Dict]:
        x_np = np.asfarray(x)
        y = ((self.alpha * x_np - 2) ** 2) * np.sin(self.beta * x_np - 4)
        return [dict(name="forrester", type="objective", value=y)]

    def get_search_space(self) -> Dict[str, str]:
        return {
            "x": "uniform(0.0, 1.0, discrete=False)",
        }
