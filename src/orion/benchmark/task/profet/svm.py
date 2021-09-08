""" Simulated Task consisting in training a Support Vector Machine (SVM).

The task is created using the Profet algorithm:

```
@misc{klein2019metasurrogate,
      title={Meta-Surrogate Benchmarking for Hyperparameter Optimization}, 
      author={Aaron Klein and Zhenwen Dai and Frank Hutter and Neil Lawrence and Javier Gonzalez},
      year={2019},
      eprint={1905.12982},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Dict, Union

import torch
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask


@dataclass
class SvmTaskHParams:
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
        input_dir: Union[Path, str] = None,
        checkpoint_dir: Union[Path, str] = None,
        train_config: MetaModelTrainingConfig = None,
        device: Union[torch.device, str] = None,
        benchmark: str = "svm",
    ):
        super().__init__(
            max_trials=max_trials,
            task_id=task_id,
            seed=seed,
            input_dir=input_dir,
            checkpoint_dir=checkpoint_dir,
            train_config=train_config,
            device=device,
            benchmark=benchmark,
        )

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))",
            gamma="loguniform(np.exp(-10), np.exp(10))",
        )
