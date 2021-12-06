""" Simulated Task consisting in training an Extreme-Gradient Boosting (XGBoost) predictor.

The task is created using the Profet algorithm. 
For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate benchmarking for 
hyperparameter optimization." Advances in Neural Information Processing Systems 32 (2019): 6270-6280.
"""
from pathlib import Path
from typing import Dict, TypedDict, Union

import torch
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class XgBoostTaskHParams(TypedDict):
    """Inputs to the XgBoost task."""

    learning_rate: float
    gamma: float
    l1_regularization: float
    l2_regularization: float
    nb_estimators: int
    subsampling: float
    max_depth: int
    min_child_weight: int


class XgBoostTask(ProfetTask[XgBoostTaskHParams]):
    """Simulated Task consisting in fitting a Extreme-Gradient Boosting predictor."""

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
        return "xgboost"

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            learning_rate="loguniform(1e-6, 1e-1)",
            gamma="uniform(0, 2, discrete=False)",
            l1_regularization="loguniform(1e-5, 1e3)",
            l2_regularization="loguniform(1e-5, 1e3)",
            nb_estimators="uniform(10, 500, discrete=True)",
            subsampling="uniform(0.1, 1)",
            max_depth="uniform(1, 15, discrete=True)",
            min_child_weight="uniform(0, 20, discrete=True)",
        )
