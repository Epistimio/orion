from typing import Any, Dict, List

import pytest
from orion.benchmark.task.base import BaseTask
from orion.benchmark.task.task_wrapper import FixTaskDimensionsWrapper, TaskWrapper
from orion.core.io.space_builder import SpaceBuilder


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
            "b": "uniform(0, 10, discrete=True)",
            "c": "uniform(0, 10, discrete=True)",
        }

    @property
    def configuration(self) -> Dict[str, Any]:
        return {
            type(self).__qualname__: dict(
                max_trials=self.max_trials, seed=self.seed, a=self.a, b=self.b, c=self.c,
            )
        }


class TestFixDimensionsWrapper:
    def test_init(self):

        original_task = DumbTask(max_trials=10)
        wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"c": 1.0})

        get_y = lambda results: results[0]["value"]

        assert get_y(original_task({"a": 1, "b": 1, "c": 1})) == 111
        assert get_y(wrapped_task({"a": 1, "b": 1, "c": 1})) == 111

        assert get_y(original_task({"a": 1, "b": 1, "c": 3})) == 113
        assert get_y(wrapped_task({"a": 1, "b": 1, "c": 3})) == 111

    def test_fixed_dim_needs_to_be_part_of_space(self):
        with pytest.raises(ValueError):
            original_task = DumbTask(10)
            wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"foo": 1.0})

    @pytest.mark.skip(reason="Not sure this makes sense. ")
    def test_space_samples_also_have_fixed_value(self):
        """Should tasks have spaces? Should the space be constrained for the fixed dimensions? """
        original_task = DumbTask(10)
        wrapped_task = FixTaskDimensionsWrapper(original_task, fixed_dims={"c": 1.0})

        for _ in range(10):
            sample = wrapped_task.space.sample()
            assert sample["c"] == 1.0

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
        new_wrapper = _get_task(
            name=name, **config_dict
        )
        assert new_wrapper.configuration == wrapper.configuration
