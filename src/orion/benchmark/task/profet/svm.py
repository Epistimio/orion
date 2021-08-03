from dataclasses import dataclass
from typing import ClassVar, Type

import numpy as np

from warmstart import HyperParameters, loguniform, uniform

from .profet_task import ProfetTask


@dataclass
class SvmTaskHParams(HyperParameters):
    C: float = loguniform(np.exp(-10), np.exp(10))
    gamma: float = loguniform(np.exp(-10), np.exp(10))


class SvmTask(ProfetTask):
    """ Simulated Task consisting in training a Support Vector Machine. """

    hparams: ClassVar[Type[SvmTaskHParams]] = SvmTaskHParams

    def __init__(self, input_path="profet_outputs", benchmark="svm", **kwargs):
        super().__init__(input_path, benchmark="svm", **kwargs)
