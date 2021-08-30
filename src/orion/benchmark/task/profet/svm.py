from dataclasses import dataclass
from typing import ClassVar, Dict, Type

from .profet_task import ProfetTask


@dataclass
class SvmTaskHParams:
    C: float
    gamma: float


class SvmTask(ProfetTask):
    """ Simulated Task consisting in training a Support Vector Machine. """
    def __init__(self, *args, benchmark="svm", **kwargs):
        super().__init__(*args, benchmark="svm", **kwargs)

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))",
            gamma="loguniform(np.exp(-10), np.exp(10))",
        )