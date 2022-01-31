""" Tests for the Quadratics Task. """

from orion.benchmark.task.quadratics import QuadraticsTask
from typing import ClassVar, Type


class TestQuadraticsTask:
    task: ClassVar[Type[QuadraticsTask]] = QuadraticsTask

    def test_seeding(self):
        seed = 123

        task_a = self.task(max_trials=123, seed=seed)
        task_b = self.task(max_trials=123, seed=seed)
        assert task_a.a_2 == task_b.a_2
        assert task_a.a_1 == task_b.a_1
        assert task_a.a_0 == task_b.a_0

        task_c = self.task(max_trials=123, seed=seed + 1)
        assert task_b.a_2 != task_c.a_2
        assert task_b.a_1 != task_c.a_1
        assert task_b.a_0 != task_c.a_0
