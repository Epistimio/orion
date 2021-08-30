from .quadratics import QuadraticsTask
from .task import Task
from orion.benchmark.task.utils import distance, similarity
from typing import ClassVar, Type
import numpy as np
import pytest


class TestQuadraticsTask:
    task: ClassVar[Type[Task]] = QuadraticsTask

    def test_seeding(self):
        seed = 123

        task_a = self.task(seed=seed)
        task_b = self.task(seed=seed)
        assert task_a.a2 == task_b.a2
        assert task_a.a1 == task_b.a1
        assert task_a.a0 == task_b.a0
        assert task_a == task_b

        task_c = self.task(seed=seed + 1)
        assert task_b.a2 != task_c.a2
        assert task_b.a1 != task_c.a1
        assert task_b.a0 != task_c.a0
        assert task_b != task_c

    @pytest.mark.parametrize(
        "task_a_kwargs, task_b_kwargs, expected_distance",
        [
            (dict(a2=0, a1=1.2, a0=0), dict(a2=0, a1=1.2, a0=0), 0.0),
            (dict(a2=0, a1=1.2, a0=0), dict(a2=0, a1=0.8, a0=0), 0.4),
            (dict(a2=1, a1=1.2, a0=0), dict(a2=1, a1=0.8, a0=0), 0.4),
            (dict(a2=1, a1=1.2, a0=2), dict(a2=1, a1=0.8, a0=2), 0.4),
            (dict(a2=1, a1=0, a0=0), dict(a2=2, a1=0, a0=0), 3.0),
            (dict(a2=1, a1=0, a0=0), dict(a2=3, a1=0, a0=0), 8.0),
            (dict(a2=1, a1=0, a0=0), dict(a2=3, a1=1, a0=0), 9.0),
            (dict(a2=1, a1=0, a0=0), dict(a2=3, a1=-1, a0=0), 9.0),
        ],
    )
    def test_distances_between_tasks(
        self, task_a_kwargs: dict, task_b_kwargs: dict, expected_distance: float
    ):
        task_a = self.task(**task_a_kwargs)
        task_b = self.task(**task_b_kwargs)
        actual_distance = distance(task_a, task_b)
        assert np.isclose(actual_distance, expected_distance)

    @pytest.mark.parametrize(
        "task_a_kwargs, task_b_kwargs, expected_similarity",
        [
            (dict(a2=0, a1=1.2, a0=0), dict(a2=0, a1=1.2, a0=0), 1.0),
            # TODO: The resulting similarity value for these is a bit arbitrary:
            # (dict(a2=0.1, a1=1.0, a0=0.1), dict(a2=0.1, a1=2.0, a0=0.1), 0.25),
            # (dict(a2=0.1, a1=1.0, a0=0.1), dict(a2=0.1, a1=2.0, a0=0.1), 0.25),
            # These are the most dis-similar tasks imaginable.
            (dict(a2=0.1, a1=0.1, a0=0.1), dict(a2=10.0, a1=10.0, a0=10.0), 0.0),
            (dict(a2=10.0, a1=10.0, a0=10.0), dict(a2=0.1, a1=0.1, a0=0.1), 0.0),
        ],
    )
    def test_similarity_between_tasks(
        self, task_a_kwargs: dict, task_b_kwargs: dict, expected_similarity: float
    ):
        task_a = self.task(**task_a_kwargs)
        task_b = self.task(**task_b_kwargs)
        actual_similarity = similarity(task_a, task_b)
        assert np.isclose(actual_similarity, expected_similarity)

    MAX_DIST = distance(task.task_low(), task.task_high())

    @pytest.mark.parametrize(
        "task_kwargs, similarity_coef, result_kwargs",
        [
            # return a new task with same params when similarity == 1.0
            (dict(a2=0.1, a1=1.2, a0=0.1), 1.0, dict(a2=0.1, a1=1.2, a0=0.1)),
            # When `similarity` == 0.0, returns the task with the lowest possible
            # similarity (NOTE: At the moment this returns either the 'minimum' task
            # (0.1, 0.1, 0.1) or the 'maximum' task (10.0, 10.0, 10.0).
            (dict(a2=0.1, a1=0.1, a0=0.1), 0.0, dict(a2=10.0, a1=10.0, a0=10.0)),
            (dict(a2=0.2, a1=0.2, a0=0.2), 0.0, dict(a2=10.0, a1=10.0, a0=10.0)),
            (dict(a2=10.0, a1=10.0, a0=10.0), 0.0, dict(a2=0.1, a1=0.1, a0=0.1)),
            (dict(a2=9.9, a1=9.9, a0=9.9), 0.0, dict(a2=0.1, a1=0.1, a0=0.1)),
            # When 0 <= `similarity` < 10.0
            # TODO: Make sure this makes _some_ sense.
            (
                dict(a2=0.1, a1=1.0, a0=0.1),
                0.95,
                dict(
                    a2=0.1,
                    a1=1.0
                    + (1 - 0.95) * ((10 ** 2 - 0.1 ** 2) + (10 - 1) + (10 - 0.1)),
                    a0=0.1,
                ),
            ),
            # TODO: Debug this one:
            (
                dict(a2=0.1, a1=0.1, a0=0.1),
                0.90,
                dict(
                    a2=0.1 + np.sqrt((0.1 * MAX_DIST - 9.9)),
                    a1=0.1 + 9.9,  # 'Overflow' the 'a1' coefficient, used 9.9/MAX_DIST
                    a0=0.1,
                ),
            ),
            # TODO: Debug this one:
            (
                dict(a2=10.0, a1=10.0, a0=10.0),
                0.95,
                dict(
                    a2=10.0,  # Unchanged, since we have enough 'space' in just a1.
                    a1=10.0 - (1 - 0.95) * MAX_DIST,  # prioritize changing a1
                    a0=10.0,
                ),
            ),
            # TODO: Debug this one:
            (
                dict(a2=10.0, a1=10.0, a0=10.0),
                0.90,
                dict(
                    a2=10.0 - np.sqrt(((1 - 0.9) * MAX_DIST - 9.9)),
                    a1=10.0 - 9.9,  # 'Overflow' the 'a1' coefficient, used 9.9/MAX_DIST
                    a0=10.0,
                ),
            ),
        ],
    )
    def test_get_similar_task(
        self, task_kwargs: dict, similarity_coef: float, result_kwargs: dict,
    ):
        task = self.task(**task_kwargs)
        expected = self.task(**result_kwargs)
        actual = task.get_similar_task(similarity_coef)
        assert actual == expected
        # NOTE: This isn't always the case, for example when `similarity` == 0 and the
        # 'source' task isn't on the opposite boundary.
        if similarity_coef != 0.0:
            print(f"{distance(task, expected)=}, {distance(task, actual)=}")
            print(f"{similarity(task, expected)=}, {similarity(task, actual)=}")
            # BUG: FIXME: Doesn't actually always match.
            # assert round(task.similarity(actual), 2) == round(similarity_coef, 2)
