""" Tests for the Quadratics Task. """
import pytest
from typing import ClassVar, Type
from orion.benchmark.task.base import BenchmarkTask
from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder
from orion.benchmark.task.quadratics import QuadraticsTask


def _get_space(task: BenchmarkTask) -> Space:
    return SpaceBuilder().build(task.get_search_space())


class TestQuadraticsTask:
    Task: ClassVar[Type[QuadraticsTask]] = QuadraticsTask

    def test_seeding(self):
        seed = 123

        task_a = self.Task(max_trials=123, seed=seed)
        task_b = self.Task(max_trials=123, seed=seed)
        assert task_a.a_2 == task_b.a_2
        assert task_a.a_1 == task_b.a_1
        assert task_a.a_0 == task_b.a_0

        task_c = self.Task(max_trials=123, seed=seed + 1)
        assert task_b.a_2 != task_c.a_2
        assert task_b.a_1 != task_c.a_1
        assert task_b.a_0 != task_c.a_0

    def test_get_search_space(self):
        task = self.Task(max_trials=123, seed=123)
        search_space_dict = task.get_search_space()
        assert set(search_space_dict.keys()) == {"x_0", "x_1", "x_2"}
        # Always the same:
        assert task.get_search_space() == search_space_dict

        task = self.Task(max_trials=123, seed=123, with_context=True)
        search_space_dict = task.get_search_space()
        assert set(search_space_dict.keys()) == {"x_0", "x_1", "x_2", "a_2", "a_1", "a_0"}
        # Always the same:
        assert task.get_search_space() == search_space_dict

    def test_call_ignores_coefficients(self):
        task = self.Task(max_trials=123, seed=123, with_context=True)
        space = _get_space(task)
        trials = space.sample(n_samples=10)
        for trial in trials:
            a_2: float = trial.params["a_2"]
            a_1: float = trial.params["a_1"]
            a_0: float = trial.params["a_0"]
            kwargs = trial.params

            kwargs_with_different_coefficients = kwargs.copy()
            kwargs_with_different_coefficients.update(a_2=2 * a_2, a_1=-4213 * a_1, a_0=a_0 + 123)

            kwargs_without_coefficients = kwargs.copy()
            kwargs_without_coefficients.pop("a_2")
            kwargs_without_coefficients.pop("a_1")
            kwargs_without_coefficients.pop("a_0")

            assert (
                task(**kwargs)
                == task(**kwargs_with_different_coefficients)
                == task(**kwargs_without_coefficients)
            )

    @pytest.mark.parametrize("a_0", [-2, 0, 2, 3])
    @pytest.mark.parametrize("with_context", [False, True])
    def test_call_a_0(self, a_0: float, with_context: bool):
        task = self.Task(max_trials=10, a_0=a_0, a_1=0.0, a_2=0.0, with_context=with_context)
        # BUG: sample() doesn't seem to respect the precision of the space, gives back full floats.
        trials = _get_space(task).sample(n_samples=5)
        for trial in trials:
            y_dicts = task(**trial.params)
            expected_y = task.a_0
            assert y_dicts == [{"name": "quadratics", "type": "objective", "value": expected_y}]

    @pytest.mark.parametrize("a_1", [-2, 0, 2, 3])
    @pytest.mark.parametrize("with_context", [False, True])
    def test_call_a_1(self, a_1: float, with_context: bool):
        task = self.Task(max_trials=10, a_0=0.0, a_1=a_1, a_2=0.0, with_context=with_context)
        trials = _get_space(task).sample(n_samples=5)
        for trial in trials:
            params = trial.params
            x_0: float = round(params["x_0"], 4)
            x_1: float = round(params["x_1"], 4)
            x_2: float = round(params["x_2"], 4)
            y_dicts = task(**params)
            expected_y = task.a_1 * (x_0 + x_1 + x_2)
            assert y_dicts == [{"name": "quadratics", "type": "objective", "value": expected_y}]

    @pytest.mark.parametrize("a_2", [-2, 0, 2, 3])
    @pytest.mark.parametrize("with_context", [False, True])
    def test_call_a_2(self, a_2: float, with_context: bool):
        task = self.Task(max_trials=10, a_0=0.0, a_1=0.0, a_2=a_2, with_context=with_context)
        trials = _get_space(task).sample(n_samples=5)
        for trial in trials:
            params = trial.params
            x_0: float = round(params["x_0"], 4)
            x_1: float = round(params["x_1"], 4)
            x_2: float = round(params["x_2"], 4)
            y_dicts = task(**params)
            expected_y = 0.5 * task.a_2 * (x_0 ** 2 + x_1 ** 2 + x_2 ** 2)
            assert y_dicts == [{"name": "quadratics", "type": "objective", "value": expected_y}]
