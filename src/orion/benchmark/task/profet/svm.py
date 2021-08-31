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
from typing import Dict

from .profet_task import ProfetTask


@dataclass
class SvmTaskHParams:
    """ Inputs to the SVM Task."""

    C: float
    gamma: float


class SvmTask(ProfetTask[SvmTaskHParams]):
    """ Simulated Task consisting in training a Support Vector Machine. """

    def __init__(self, *args, benchmark="svm", **kwargs):
        super().__init__(*args, benchmark=benchmark, **kwargs)

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))",
            gamma="loguniform(np.exp(-10), np.exp(10))",
        )
