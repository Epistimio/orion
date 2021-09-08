""" Simulated Task consisting in training a fully-connected network. 

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
from pathlib import Path
from typing import Dict, Union
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask


@dataclass
class FcNetTaskHParams:
    """Hyper-Parameters of a Simulated Task consisting in training a fully-connected network."""

    learning_rate: float
    batch_size: int
    units_layer1: int
    units_layer2: int
    dropout_rate_l1: float
    dropout_rate_l2: float


class FcNetTask(ProfetTask[FcNetTaskHParams]):
    """Simulated Task consisting in training a fully-connected network."""

    def __init__(
        self,
        max_trials: int = 100,
        task_id: int = 0,
        seed: int = 123,
        input_dir: Union[Path, str] = None,
        checkpoint_dir: Union[Path, str] = None,
        train_config: MetaModelTrainingConfig = None,
        device: Union[torch.device, str] = None,
        benchmark: str = "fcnet",
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
            learning_rate="loguniform(1e-6, 1e-1)",
            batch_size="loguniform(8, 128, discrete=True)",
            units_layer1="loguniform(16, 512, discrete=True)",
            units_layer2="loguniform(16, 512, discrete=True)",
            dropout_rate_l1="uniform(0, 0.99)",
            dropout_rate_l2="uniform(0, 0.99)",
        )
