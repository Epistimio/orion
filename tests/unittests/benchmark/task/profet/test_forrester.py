""" Tests for the Forrester task. """
from typing import ClassVar, Type

from orion.benchmark.task.profet.forrester import ProfetForresterTask
from orion.benchmark.task.profet.profet_task import ProfetTask

from .test_profet_task import ProfetTaskTests


class TestForresterTask(ProfetTaskTests):
    """Tests for the `ForresterTask` class."""

    Task: ClassVar[Type[ProfetTask]] = ProfetForresterTask
