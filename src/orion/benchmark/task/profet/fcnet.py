""" Simulated Task consisting in training a fully-connected network.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

from orion.benchmark.task.profet.profet_task import ProfetTask


class ProfetFcNetTask(ProfetTask):
    """Simulated Task consisting in training a fully-connected network."""

    @dataclass
    class ModelConfig(ProfetTask.ModelConfig):
        """Config for training the Profet model on an FcNet task."""

        benchmark: str = "fcnet"
        json_file_name: ClassVar[str] = "data_sobol_fcnet.json"
        hidden_space: ClassVar[int] = 5
        log_cost: ClassVar[bool] = True
        log_target: ClassVar[bool] = False
        normalize_targets: ClassVar[bool] = False

        shapes: ClassVar[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = (
            (600, 6),
            (27, 600),
            (27, 600),
        )
        y_min: ClassVar[float] = 0.0
        y_max: ClassVar[float] = 1.0
        c_min: ClassVar[float] = 0.0
        c_max: ClassVar[float] = 14718.31848526001

    def call(
        self,
        learning_rate: float,
        batch_size: int,
        units_layer1: int,
        units_layer2: int,
        dropout_rate_l1: float,
        dropout_rate_l2: float,
    ) -> List[Dict]:
        return super().call(
            learning_rate=learning_rate,
            batch_size=batch_size,
            units_layer1=units_layer1,
            units_layer2=units_layer2,
            dropout_rate_l1=dropout_rate_l1,
            dropout_rate_l2=dropout_rate_l2,
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
