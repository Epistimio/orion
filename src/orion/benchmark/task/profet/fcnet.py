from dataclasses import dataclass
from typing import ClassVar, Type

from warmstart import HyperParameters, loguniform, uniform

from .profet_task import ProfetTask


@dataclass
class FcNetTaskHParams(HyperParameters):
    learning_rate: float = loguniform(1e-6, 1e-1)
    batch_size: int = loguniform(8, 128, discrete=True)
    units_layer1: int = loguniform(16, 512, discrete=True)
    units_layer2: int = loguniform(16, 512, discrete=True)
    dropout_rate_l1: float = uniform(0, 0.99)
    dropout_rate_l2: float = uniform(0, 0.99)


class FcNetTask(ProfetTask):
    """ Simulated Task consisting in training a fully-connected network. """

    hparams: ClassVar[Type[FcNetTaskHParams]] = FcNetTaskHParams

    def __init__(self, input_path, benchmark="fcnet", task_idx: int = None, **kwargs):
        super().__init__(input_path, benchmark="fcnet", task_idx=task_idx, **kwargs)
