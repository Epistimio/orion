from abc import ABC
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    TypeVar,
    Union,
)
import copy

import numpy as np
from orion.algo.space import Dimension, Space
from orion.benchmark.task.base import BaseTask, BenchmarkTask
from logging import getLogger as get_logger

from orion.core.utils.points import flatten_dims, regroup_dims
from orion.core.utils.format_trials import dict_to_trial

from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_trials import trial_to_tuple, tuple_to_trial
from orion.core.worker.trial import Trial

logger = get_logger(__name__)


TaskType = TypeVar("TaskType", bound=BaseTask)
InputType = TypeVar("InputType")


class TaskWrapper(BaseTask, Generic[TaskType], ABC):
    """ ABC for a Wrapper around a Task.

    Wrappers could be used to modify the search space, trials, or the results of the task.
    
    Parameters
    ----------
    task : TaskType
        A task to wrap.
    max_trials : int, optional
        Maximum number of trials. By default None, in which case the max trials is the same as that
        of the wrapped task.
    """

    def __init__(self, task: TaskType, max_trials: int = None):
        # TODO: Should Wrappers pass the `task` as an argument to the super() constructor?
        # When/How are these objects deserialized?
        if isinstance(task, dict):
            if len(task) != 1:
                raise ValueError(
                    f"Expected task configuration dict to have only one key (the task name), but "
                    f"got {task} instead."
                )
            task_name, task_config = task.popitem()
            task = BenchmarkTask(of_type=task_name, **task_config)
        super().__init__(max_trials=max_trials or task.max_trials)
        self.task: TaskType = task

    def call(self, x: Any) -> List[Dict]:
        return self.task.call(x)

    def get_search_space(self) -> Dict[str, str]:
        return self.task.get_search_space()

    @property
    def unwrapped(self) -> Union[TaskType, BaseTask]:
        """ Returns the 'unwrapped' task. """
        return self.task.unwrapped

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task wrapper (including the wrapped task's configuration).
        """
        return {
            type(self).__qualname__: {
                "task": self.task.configuration,
                "max_trials": self.max_trials,
            }
        }


HParamsType = TypeVar("HParamsType")


class FixTaskDimensionsWrapper(TaskWrapper[TaskType]):
    """ Wrapper around a Task that fixes the values of some of its input dimensions.

    Parameters
    ----------
    task : TaskType
        A task to wrap.
    fixed_dims : Dict[str, Any]
        Dictionary mapping from the name of a dimension of the task space to the value that
        dimension should be fixed at.
    max_trials : int, optional
        Maximum number of trials, by default None
    """

    def __init__(self, task: TaskType, fixed_dims: Dict[str, Any], max_trials: int = None):
        super().__init__(task=task, max_trials=max_trials)
        self.fixed_dims = fixed_dims
        # The whole space, from the wrapped task.
        self._full_space: Space = SpaceBuilder().build(self.task.get_search_space())

        for dimension_name in fixed_dims:
            if dimension_name not in self._full_space:
                raise ValueError(
                    f"Can't fix dimension '{dimension_name}' because it isn't in the task's space: "
                    f"{self._full_space}."
                )

        # The part of the space that can be modified.
        self._space: Space = SpaceBuilder().build(self.get_search_space())

    def call(self, x: Any) -> List[Dict]:
        """ Calls the wrapped task, adding/changing some dimensions in `x` if necessary. """
        trial: Trial
        if isinstance(x, np.ndarray):
            point_tuple = regroup_dims(x, space=self._full_space)
            trial = tuple_to_trial(point_tuple, space=self._full_space)
        elif isinstance(x, dict):
            trial = dict_to_trial(x, space=self._full_space)
        elif isinstance(x, Trial):
            trial = copy.deepcopy(trial)
        else:
            raise NotImplementedError(
                f"Don't know how to fix some values for input {trial} of type {type(trial)}"
            )

        trial_was_modified = False
        for name, fixed_value in self.fixed_dims.items():
            params = trial.params

            if name not in params:
                # The trial doesn't have one of the values we want fixed: Set it in the trial.
                dim: Dimension = self._full_space[name]
                # pylint: disable=protected-access
                trial._params.append(Trial.Value(name=name, _type=dim.type, value=fixed_value))
                trial_was_modified = True

            elif params[name] != fixed_value:
                # pylint: disable=protected-access
                offending_value: Trial.Value = [
                    value for value in trial._params if value.name == name
                ][0]
                offending_value.value = fixed_value
                trial_was_modified = True
                # Change should be reflected immediately since we're modifying the Value in-place.
                assert trial.params[name] == fixed_value

        if not trial_was_modified:
            # Can return `x` directly, since we didn't end up modifying it.
            new_x = x
        elif isinstance(x, np.ndarray):
            point_tuple = trial_to_tuple(trial, self._space)
            flattened_point = flatten_dims(point_tuple, self._space)
            new_x = np.array(flattened_point)
        elif isinstance(x, dict):
            new_x = trial.params
        elif isinstance(x, Trial):
            # NOTE: The modifications were performed in-place with respect to the copy.
            new_x = trial

        return super().call(new_x)

    def get_search_space(self) -> Dict[str, str]:
        """ Returns the truncated search space (without the fixed values). """
        original_space = super().get_search_space()
        modified_space = original_space.copy()
        for dimension_name in self.fixed_dims.keys():
            modified_space.pop(dimension_name)
        return modified_space

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task wrapper.
        """
        config = super().configuration
        config[type(self).__qualname__]["fixed_dims"] = self.fixed_dims.copy()
        return config
