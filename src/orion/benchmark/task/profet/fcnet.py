""" Simulated Task consisting in training a fully-connected network.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List

from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from torch import nn

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


class ProfetFcNetTask(ProfetTask):
    """Simulated Task consisting in training a fully-connected network."""

    @dataclass
    class ModelConfig(MetaModelConfig):
        """Config for training the Profet model on an FcNet task."""

        benchmark: Final[str] = "fcnet"
        # ---------- "Abstract" class attributes:
        json_file_name: ClassVar[str] = "data_sobol_fcnet.json"
        get_architecture: ClassVar[Callable[[int], nn.Module]] = partial(get_default_architecture)
        hidden_space: ClassVar[int] = 5
        log_cost: ClassVar[bool] = True
        log_target: ClassVar[bool] = False
        normalize_targets: ClassVar[bool] = False
        # -----------

    def call(
        self,
        *args,
        learning_rate: float,
        batch_size: int,
        units_layer1: int,
        units_layer2: int,
        dropout_rate_l1: float,
        dropout_rate_l2: float,
        **kwargs,
    ) -> List[Dict]:
        return super().call(
            *args,
            learning_rate=learning_rate,
            batch_size=batch_size,
            units_layer1=units_layer1,
            units_layer2=units_layer2,
            dropout_rate_l1=dropout_rate_l1,
            dropout_rate_l2=dropout_rate_l2,
            **kwargs,
        )

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            learning_rate="loguniform(1e-6, 1e-1)",
            batch_size="loguniform(8, 128, discrete=True)",
            units_layer1="loguniform(16, 512, discrete=True)",
            units_layer2="loguniform(16, 512, discrete=True)",
            dropout_rate_l1="uniform(0, 0.99)",
            dropout_rate_l2="uniform(0, 0.99)",
        )
