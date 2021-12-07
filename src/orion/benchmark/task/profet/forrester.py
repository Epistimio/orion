""" Hyper-Parameters of a simulated task based on variants of the Forrester function.
"""
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, ClassVar, Dict, Union

import torch
from emukit.examples.profet.meta_benchmarks.meta_forrester import (
    get_architecture_forrester,
)
from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from torch import nn

logger = get_logger(__name__)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ForresterTaskHParams(TypedDict):
    """Hyper-Parameters of a simulated task based on variants of the Forrester function:

    $ f(x) = ((\alpha x - 2)^2) sin(\beta x - 4) $

    NOTE: There appears to be a discrepancy between the paper's description of this task (with two
    parameters \alpha and \beta in [0,1]) and the code implementation at
    https://github.com/EmuKit/emukit/blob/main/emukit/examples/profet/meta_benchmarks/meta_forrester.py
    where the latter has a single `x` parameter in the interval [0,1].

    TODO: Run this with the real data and check which of the two space definitions matches the data.
    """

    alpha: float
    beta: float

    # x: float


class ForresterTask(ProfetTask[ForresterTaskHParams]):
    """Simulated Task consisting in training a model on a variant of the Forrester function."""

    @dataclass
    class ModelConfig(MetaModelConfig):
        """ Config for training the Profet model on a Forrester task. """

        # ---------- "Abstract" class attributes:
        benchmark: ClassVar[str] = "forrester"
        json_file_name: ClassVar[str] = "data_sobol_forrester.json"
        get_architecture: ClassVar[
            Callable[[int], nn.Module]
        ] = get_architecture_forrester
        hidden_space: ClassVar[int] = 2
        normalize_targets: ClassVar[bool] = True
        log_cost: ClassVar[bool] = False
        log_target: ClassVar[bool] = False
        # -----------

    def get_search_space(self) -> Dict[str, str]:
        return {
            "alpha": "uniform(0.0, 1.0, discrete=False)",
            "beta": "uniform(0.0, 1.0, discrete=False)",
            # "x": "uniform(0.0, 1.0, discrete=False)",
        }
