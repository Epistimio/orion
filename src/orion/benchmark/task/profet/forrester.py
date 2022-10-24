"""Hyper-Parameters of a simulated task based on variants of the Forrester function:

.. math:: f(x) = ((\alpha x - 2)^2) sin(\beta x - 4)


This task uses a meta-model that is trained using a dataset of points from different functions, each
with different values of alpha and beta. This meta-model can then be used to sample "fake" points
from a given forrester function.
"""
import typing
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Tuple

from orion.benchmark.task.profet.model_utils import get_architecture_forrester
from orion.benchmark.task.profet.profet_task import ProfetTask

if typing.TYPE_CHECKING:
    import torch


class ProfetForresterTask(ProfetTask):
    """Simulated Task consisting in training a model on a variant of the Forrester function."""

    @dataclass
    class ModelConfig(ProfetTask.ModelConfig):
        """Config for training the Profet model on a Forrester task."""

        benchmark: str = "forrester"

        # ---------- "Abstract" class attributes:
        json_file_name: ClassVar[str] = "data_sobol_forrester.json"
        get_architecture: ClassVar[
            Callable[[int], "torch.nn.Module"]
        ] = get_architecture_forrester
        """ Callable that takes the input dimensionality and returns the network to be trained. """
        hidden_space: ClassVar[int] = 2
        normalize_targets: ClassVar[bool] = True
        log_cost: ClassVar[bool] = False
        log_target: ClassVar[bool] = False
        shapes: ClassVar[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = (
            (10, 1),
            (9, 10),
            (9, 10),
        )
        y_min: ClassVar[float] = -18.049155413936802
        y_max: ClassVar[float] = 14718.31848526001
        c_min: ClassVar[float] = -18.049155413936802
        c_max: ClassVar[float] = 14718.3184852600
        # -----------

    def call(self, x: float) -> List[Dict]:
        return super().call(x=x)

    def get_search_space(self) -> Dict[str, str]:
        return {
            "x": "uniform(0.0, 1.0, discrete=False)",
        }
