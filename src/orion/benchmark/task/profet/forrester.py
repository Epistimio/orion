""" Simulated Task consisting in training a Random Forrest predictor.

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

from logging import getLogger as get_logger

from .profet_task import ProfetTask

logger = get_logger(__name__)


@dataclass
class ForresterTaskHParams:
    """ Hyper-Parameters of a Simulated Task consisting in training a Random Forrest predictor.
    """
    # TODO (@lebrice): This space is supposedly not correct. Need to look at
    # the profet paper in a bit more detail to check what the 'x' range is
    # supposed to be.
    # alpha='uniform(0, 1)',
    # beta='uniform(0, 1)'
    x: float


class ForresterTask(ProfetTask):
    """ Simulated Task consisting in training a Random Forrest predictor. """

    def __init__(self, *args, benchmark="forrester", **kwargs):
        super().__init__(*args, benchmark=benchmark, **kwargs)

    def get_search_space(self) -> Dict[str, str]:
        return {"x": "uniform(0.0, 1.0, discrete=False)"}
