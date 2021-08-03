from dataclasses import dataclass
from typing import ClassVar, Type

from warmstart import HyperParameters, loguniform, uniform
from logging import getLogger as get_logger

from .profet_task import ProfetTask

logger = get_logger(__name__)


@dataclass
class ForresterTaskHParams(HyperParameters):
    x: float = uniform(0, 1, discrete=False)


class ForresterTask(ProfetTask):
    """ Simulated Task consisting in training a Random Forrest predictor. """

    hparams: ClassVar[Type[ForresterTaskHParams]] = ForresterTaskHParams

    def __init__(self, input_path, benchmark="forrester", **kwargs):
        super().__init__(input_path, benchmark="forrester", **kwargs)

    def __call__(self, hp, **kwargs):
        logger.debug(
            f"hp: {hp}, kwargs: {kwargs}, self.fixed_values: {self.fixed_values}"
        )
        return super().__call__(hp, **kwargs)
