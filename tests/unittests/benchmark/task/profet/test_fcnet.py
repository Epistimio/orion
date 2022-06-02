""" Tests for the FcNet Task. """
from typing import ClassVar, Type

from orion.benchmark.task.profet.fcnet import ProfetFcNetTask
from orion.benchmark.task.profet.profet_task import ProfetTask

from .test_profet_task import ProfetTaskTests


class TestFcNetTask(ProfetTaskTests):
    """Tests for the `FcNetTask` class."""

    Task: ClassVar[Type[ProfetTask]] = ProfetFcNetTask
