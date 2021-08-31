""" Simulated Task consisting in training an Extreme-Gradient Boosting (XGBoost) predictor.

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
from typing import Dict


from .profet_task import ProfetTask


@dataclass
class XgBoostTaskHParams:
    """ Inputs to the XgBoost task. """

    learning_rate: float
    gamma: float
    l1_regularization: float
    l2_regularization: float
    nb_estimators: int
    subsampling: float
    max_depth: int
    min_child_weight: int


class XgBoostTask(ProfetTask[XgBoostTaskHParams]):
    """ Simulated Task consisting in fitting a Extreme-Gradient Boosting predictor.
    """

    def __init__(self, *args, benchmark="xgboost", **kwargs):
        super().__init__(*args, benchmark=benchmark, **kwargs)

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
