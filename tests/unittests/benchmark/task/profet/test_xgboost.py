""" Tests for the XGBoost task. """
from typing import ClassVar, Type

from orion.benchmark.task.profet.profet_task import ProfetTask
from orion.benchmark.task.profet.xgboost import ProfetXgBoostTask

from .test_profet_task import ProfetTaskTests


class TestXgBoostTask(ProfetTaskTests):
    """Tests for the XGBoostTask class."""

    Task: ClassVar[Type[ProfetTask]] = ProfetXgBoostTask
