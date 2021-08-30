""" Tests for the Quadratics Task. """

from orion.benchmark.task.quadratics import QuadraticsTask
from orion.benchmark.task.base import BaseTask
from typing import ClassVar, Type


class TestQuadraticsTask:
    task: ClassVar[Type[BaseTask]] = QuadraticsTask

    def test_seeding(self):
        seed = 123

        task_a = self.task(max_trials=123, seed=seed)
        task_b = self.task(max_trials=123, seed=seed)
        assert task_a.a2 == task_b.a2
        assert task_a.a1 == task_b.a1
        assert task_a.a0 == task_b.a0

        task_c = self.task(max_trials=123, seed=seed + 1)
        assert task_b.a2 != task_c.a2
        assert task_b.a1 != task_c.a1
        assert task_b.a0 != task_c.a0
