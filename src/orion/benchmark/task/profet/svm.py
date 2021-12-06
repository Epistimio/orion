""" Simulated Task consisting in training a Support Vector Machine (SVM).

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

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict


class SvmTaskHParams(TypedDict):
    """Inputs to the SVM Task."""

    C: float
    gamma: float


class SvmTask(ProfetTask[SvmTaskHParams]):
    """Simulated Task consisting in training a Support Vector Machine."""

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
        return "svm"

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))", gamma="loguniform(np.exp(-10), np.exp(10))",
        )
