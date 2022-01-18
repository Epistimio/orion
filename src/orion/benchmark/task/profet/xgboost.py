""" Simulated Task consisting in training an Extreme-Gradient Boosting (XGBoost) predictor.
"""
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List

from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from torch import nn

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


class XgBoostTask(ProfetTask):
    """Simulated Task consisting in fitting a Extreme-Gradient Boosting predictor."""

    @dataclass
    class ModelConfig(MetaModelConfig):
        """Config for training the Profet model on an XgBoost task."""

        benchmark: Final[str] = "xgboost"

        # ---------- "Abstract" class attributes:
        json_file_name: ClassVar[str] = "data_sobol_xgboost.json"
        get_architecture: ClassVar[Callable[[int], nn.Module]] = get_default_architecture
        hidden_space: ClassVar[int] = 5
        normalize_targets: ClassVar[bool] = True
        log_cost: ClassVar[bool] = True
        log_target: ClassVar[bool] = True
        # -----------

    def call(
        self,
        *args,
        learning_rate: float,
        gamma: float,
        l1_regularization: float,
        l2_regularization: float,
        nb_estimators: int,
        subsampling: float,
        max_depth: int,
        min_child_weight: int,
        **kwargs
    ) -> List[Dict]:
        return super().call(
            *args,
            learning_rate=learning_rate,
            gamma=gamma,
            l1_regularization=l1_regularization,
            l2_regularization=l2_regularization,
            nb_estimators=nb_estimators,
            subsampling=subsampling,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            **kwargs
        )

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
