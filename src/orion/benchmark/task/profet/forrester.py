""" Simulated Task consisting in training a Random Forrest predictor.

The task is created using the Profet algorithm. 
For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate benchmarking for 
hyperparameter optimization." Advances in Neural Information Processing Systems 32 (2019): 6270-6280.
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Dict, Union

import torch
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask

logger = get_logger(__name__)


@dataclass
class ForresterTaskHParams:
    """Hyper-Parameters of a Simulated Task consisting in training a Random Forrest predictor."""

    # NOTE: (@lebrice) This space is difference in OpenML:
    # alpha='uniform(0, 1)',
    # beta='uniform(0, 1)'
    x: float


class ForresterTask(ProfetTask[ForresterTaskHParams]):
    """Simulated Task consisting in training a Random Forrest predictor."""

    def __init__(
        self,
        max_trials: int = 100,
        task_id: int = 0,
        seed: int = 123,
        input_dir: Union[Path, str] = None,
        checkpoint_dir: Union[Path, str] = None,
        train_config: MetaModelTrainingConfig = None,
        device: Union[torch.device, str] = None,
        with_grad: bool = False,
        benchmark: str = "forrester",
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
            benchmark=benchmark,
        )

    def get_search_space(self) -> Dict[str, str]:
        return {"x": "uniform(0.0, 1.0, discrete=False)"}
