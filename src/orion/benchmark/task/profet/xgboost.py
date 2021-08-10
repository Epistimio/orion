from dataclasses import dataclass
from typing import ClassVar, Type

from simple_parsing.helpers.hparams import HyperParameters, loguniform, uniform

from .profet_task import ProfetTask


@dataclass
class XgBoostTaskHParams(HyperParameters):
    learning_rate: float = loguniform(1e-6, 1e-1)
    gamma: float = uniform(0., 2.)
    l1_regularization: float = loguniform(1e-5, 1e3)
    l2_regularization: float = loguniform(1e-5, 1e3)
    nb_estimators: int = uniform(10, 500, discrete=True)
    subsampling: float = uniform(0.1, 1)
    max_depth: int = uniform(1, 15, discrete=True)
    min_child_weight: int = uniform(0, 20, discrete=True)


class XgBoostTask(ProfetTask):
    """ Simulated Task consisting in fitting a Extreme-Gradient Boosting predictor.
    """
    hparams: ClassVar[Type[XgBoostTaskHParams]] = XgBoostTaskHParams

    def __init__(self, *args, benchmark="xgboost", **kwargs):
        super().__init__(*args, benchmark="xgboost", **kwargs)

