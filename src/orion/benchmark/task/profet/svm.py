from dataclasses import dataclass
from typing import ClassVar, Type

import numpy as np

from simple_parsing.helpers.hparams import HyperParameters, loguniform

from .profet_task import ProfetTask


@dataclass
class SvmTaskHParams(HyperParameters):
    C: float = loguniform(np.exp(-10), np.exp(10))
    gamma: float = loguniform(np.exp(-10), np.exp(10))


class SvmTask(ProfetTask):
    """ Simulated Task consisting in training a Support Vector Machine. """

    hparams: ClassVar[Type[SvmTaskHParams]] = SvmTaskHParams

    def __init__(self, *args, benchmark="svm", **kwargs):
        super().__init__(*args, benchmark="svm", **kwargs)
