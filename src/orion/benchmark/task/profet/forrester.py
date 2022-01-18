""" Hyper-Parameters of a simulated task based on variants of the Forrester function:

.. math::
f(x) = ((\\alpha x - 2)^2) sin(\\beta x - 4)
..

This task uses a meta-model that is trained using a dataset of points from different functions, each
with different values of alpha and beta. This meta-model can then be used to sample "fake" points
from a given forrester function.

TODO: Is there really a use for this? The "base" task is already a synthetic task, why would we have
a meta-model for a synthetic task, if we can just run the original task very quickly & efficiently?
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable, ClassVar, Dict, List

from emukit.examples.profet.meta_benchmarks.meta_forrester import get_architecture_forrester
from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from torch import nn

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final  # type: ignore

logger = get_logger(__name__)


class ForresterTask(ProfetTask):
    """Simulated Task consisting in training a model on a variant of the Forrester function. """

    @dataclass
    class ModelConfig(MetaModelConfig):
        """Config for training the Profet model on a Forrester task."""

        benchmark: Final[str] = "forrester"

        # ---------- "Abstract" class attributes:
        json_file_name: ClassVar[str] = "data_sobol_forrester.json"
        get_architecture: ClassVar[Callable[[int], nn.Module]] = get_architecture_forrester
        hidden_space: ClassVar[int] = 2
        normalize_targets: ClassVar[bool] = True
        log_cost: ClassVar[bool] = False
        log_target: ClassVar[bool] = False
        # -----------

    def call(self, *args, x: float, **kwargs) -> List[Dict]:
        return super().call(*args, x=x, **kwargs)

    def get_search_space(self) -> Dict[str, str]:
        return {
            "x": "uniform(0.0, 1.0, discrete=False)",
        }

