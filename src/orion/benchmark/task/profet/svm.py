""" Simulated Task consisting in training a Support Vector Machine (SVM).
"""
import typing
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List, Tuple

from orion.benchmark.task.profet.model_utils import get_default_architecture
from orion.benchmark.task.profet.profet_task import ProfetTask

if typing.TYPE_CHECKING:
    import torch


class ProfetSvmTask(ProfetTask):
    """Simulated Task consisting in training a Support Vector Machine."""

    @dataclass
    class ModelConfig(ProfetTask.ModelConfig):
        """Config for training the Profet model on an SVM task."""

        benchmark: str = "svm"

        json_file_name: ClassVar[str] = "data_sobol_svm.json"
        get_architecture: ClassVar[Callable[[int], "torch.nn.Module"]] = partial(
            get_default_architecture, classification=True
        )
        """ Callable that takes the input dimensionality and returns the network to be trained. """

        hidden_space: ClassVar[int] = 5
        normalize_targets: ClassVar[bool] = False
        log_cost: ClassVar[bool] = True
        log_target: ClassVar[bool] = False
        shapes: ClassVar[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = (
            (200, 2),
            (26, 200),
            (26, 200),
        )
        y_min: ClassVar[float] = 0.0
        y_max: ClassVar[float] = 1.0
        c_min: ClassVar[float] = 0.0
        c_max: ClassVar[float] = 697154.4010462761

    def call(self, C: float, gamma: float) -> List[Dict]:
        return super().call(C=C, gamma=gamma)

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))",
            gamma="loguniform(np.exp(-10), np.exp(10))",
        )
