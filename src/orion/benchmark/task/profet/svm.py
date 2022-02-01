""" Simulated Task consisting in training a Support Vector Machine (SVM).
"""
import typing
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List
from orion.benchmark.task.profet.profet_task import ProfetTask
from orion.benchmark.task.profet.model_utils import MetaModelConfig, get_default_architecture

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final  # type: ignore

if typing.TYPE_CHECKING:
    from torch import nn


class ProfetSvmTask(ProfetTask):
    """Simulated Task consisting in training a Support Vector Machine."""

    @dataclass
    class ModelConfig(MetaModelConfig):
        """Config for training the Profet model on an SVM task."""

        benchmark: Final[str] = "svm"

        # ---------- "Abstract" class attributes:
        json_file_name: ClassVar[str] = "data_sobol_svm.json"
        get_architecture: ClassVar[Callable[[int], "nn.Module"]] = partial(
            get_default_architecture, classification=True
        )
        hidden_space: ClassVar[int] = 5
        normalize_targets: ClassVar[bool] = False
        log_cost: ClassVar[bool] = True
        log_target: ClassVar[bool] = False
        # -----------

    def call(self, C: float, gamma: float) -> List[Dict]:
        return super().call(C=C, gamma=gamma)

    def get_search_space(self) -> Dict[str, str]:
        return dict(
            C="loguniform(np.exp(-10), np.exp(10))", gamma="loguniform(np.exp(-10), np.exp(10))",
        )
