""" Tests for the SVM task. """
from typing import ClassVar, Type

from orion.benchmark.task.profet.profet_task import ProfetTask
from orion.benchmark.task.profet.svm import ProfetSvmTask

from .test_profet_task import ProfetTaskTests


class TestSvmTask(ProfetTaskTests):
    """Tests for the `SvmTask` class."""

    Task: ClassVar[Type[ProfetTask]] = ProfetSvmTask
