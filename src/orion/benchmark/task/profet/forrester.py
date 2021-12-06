""" Hyper-Parameters of a simulated task based on variants of the Forrester function.
"""
from logging import getLogger as get_logger
from pathlib import Path
from typing import Dict, Union

import torch
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask

logger = get_logger(__name__)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ForresterTaskHParams(TypedDict):
    """Hyper-Parameters of a simulated task based on variants of the Forrester function:

    $ f(x) = ((\alpha x - 2)^2) sin(\beta x - 4) $

    NOTE: There appears to be a discrepancy between the paper's description of this task (with two
    parameters \alpha and \beta in [0,1]) and the code implementation at
    https://github.com/EmuKit/emukit/blob/main/emukit/examples/profet/meta_benchmarks/meta_forrester.py
    where the latter has a single `x` parameter in the interval [0,1].

    TODO: Run this with the real data and check which of the two space definitions matches the data.
    """
    alpha: float
    beta: float

    # x: float


class ForresterTask(ProfetTask[ForresterTaskHParams]):
    """Simulated Task consisting in training a Random Forrest predictor."""

    def __init__(
        self,
        max_trials: int = 100,
        task_id: int = 0,
        seed: int = 123,
        input_dir: Union[Path, str] = "profet_data",
        checkpoint_dir: Union[Path, str] = None,
        train_config: MetaModelTrainingConfig = None,
        device: Union[torch.device, str] = None,
        with_grad: bool = False,
    ):
        super().__init__(
            max_trials=max_trials,
            task_id=task_id,
            seed=seed,
            input_dir=input_dir,
            checkpoint_dir=checkpoint_dir,
            train_config=train_config,
            device=device,
            with_grad=with_grad,
        )

    @property
    def benchmark(self) -> str:
        """ The name of the benchmark to use. """
        return "forrester"

    def get_search_space(self) -> Dict[str, str]:
        return {
            "alpha": "uniform(0.0, 1.0, discrete=False)",
            "beta": "uniform(0.0, 1.0, discrete=False)",
            # "x": "uniform(0.0, 1.0, discrete=False)",
        }
