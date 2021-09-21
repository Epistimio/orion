from typing import Any, Dict, List

import pytest
from orion.benchmark.task.base import BaseTask
from orion.benchmark.task.task_wrapper import FixTaskDimensionsWrapper


class DumbTask(BaseTask):
    def __init__(self, max_trials: int, seed: int = None, a: int = 100, b: int = 10, c: int = 1):
        super().__init__(max_trials=max_trials)
        self.a = a
        self.b = b
        self.c = c
        self.seed = seed

    def call(self, x: Dict) -> List[Dict]:
        y = self.a * x["a"] + self.b * x["b"] + self.c * x["c"]
        return [dict(name="dumb_task", type="objective", value=y)]

    def get_search_space(self) -> Dict[str, str]:
        return {
            "a": "uniform(0, 10, discrete=True)",
            "b": "uniform(0, 11, discrete=True)",
            "c": "uniform(0, 12, discrete=True)",
        }

    @property
    def configuration(self) -> Dict[str, Any]:
        return {
            type(self).__qualname__: dict(
                max_trials=self.max_trials, seed=self.seed, a=self.a, b=self.b, c=self.c,
            )
        }


class TestFixDimensionsWrapper:
    @pytest.mark.parametrize("task_as_dict", [True, False])
    def test_init(self, task_as_dict: bool):
        original_task = DumbTask(max_trials=10)
        if task_as_dict:
            wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"c": 1.0})
        else:
            wrapped_task = FixTaskDimensionsWrapper(
                original_task.configuration, fixed_dims={"c": 1.0}
            )

        get_y = lambda results: results[0]["value"]

        assert get_y(original_task({"a": 1, "b": 1, "c": 1})) == 111
        assert get_y(wrapped_task({"a": 1, "b": 1, "c": 1})) == 111

        assert get_y(original_task({"a": 1, "b": 1, "c": 3})) == 113
        assert get_y(wrapped_task({"a": 1, "b": 1, "c": 3})) == 111

    def test_get_search_space(self):
        original_task = DumbTask(max_trials=10)
        wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"c": 1.0})
        expected_space = original_task.get_search_space().copy()
        _ = expected_space.pop("c")
        assert wrapped_task.get_search_space() == expected_space

    def test_fixed_dim_needs_to_be_part_of_space(self):
        with pytest.raises(
            ValueError, match="Can't fix dimension 'foo' because it isn't in the task's space"
        ):
            original_task = DumbTask(10)
            wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"foo": 1.0})

    def test_configuration(self):
        """ Test that the configuration dict contains the config of the wrapped task. """
        task = DumbTask(max_trials=10, a=5, b=4, c=3)
        task_config = task.configuration
        wrapper = FixTaskDimensionsWrapper(task, max_trials=123, fixed_dims={"c": 1})

        from orion.benchmark.benchmark_client import _get_task

        assert wrapper.configuration == {
            FixTaskDimensionsWrapper.__qualname__: {
                "task": task_config,
                "max_trials": 123,
                "fixed_dims": {"c": 1},
            }
        }

        # Test de-serializing a configuration dict for a task wrapper into a wrapper
        name, config_dict = wrapper.configuration.copy().popitem()
        new_wrapper = _get_task(name=name, **config_dict)
        assert new_wrapper.configuration == wrapper.configuration
