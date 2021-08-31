from abc import ABC
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Tuple,
    TypeVar,
    Union,
)
import copy

import numpy as np
from orion.algo.space import Dimension, Space
from orion.analysis.base import flatten_params
from orion.benchmark.task.base import BaseTask
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
    """

    def __init__(self, task: TaskType, max_trials: int = None):
        # TODO: Should Wrappers pass the `task` as an argument to the super() constructor?
        # When/How are these objects deserialized?
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
        """Return the configuration of the task."""
        # TODO: Figure this out (see above TODO about passing `task` to the constructor.
        # return {self.__class__.__qualname__: {"task": self.task.configuration}}
        # IDEA: Maybe use something like this to get the configuration from the signature?
        from inspect import signature, Signature

        init_signature: Signature = signature(type(self).__init__)
        return {
            self.__class__.__qualname__: {
                name: getattr(self, name, parameter.default)
                for name, parameter in init_signature.parameters.items()
            }
        }


HParamsType = TypeVar("HParamsType")


class AcceptDataclassesWrapper(TaskWrapper, Generic[TaskType, HParamsType]):
    """ Wrapper around a task, makes it so that task can accept dataclasses as well as dicts.

    Can evaluate dataclasses rather than just dicts.
    """

    def call(self, x: Union[HParamsType, Any]) -> List[Dict]:
        """Evaluates the given samples and returns the performance.

        Args:
            hp (Union[HyperParameters, Dict, np.ndarray], optional):
                Either a Hyperparameter dataclass, a dictionary, or a numpy
                array containing the values for each dimension. Defaults to
                None.

        Returns:
            List[Dict] The performances of the hyperparameter on the sampled
            task.
        """
        if is_dataclass(x):
            x = asdict(x)
        return self.task.call(x)


class CanReturnFloatWrapper(TaskWrapper[TaskType]):
    """ Wrapper that allows tasks to simply return floats rather than `Results` objects. """

    def __init__(self, task: TaskType, max_trials: int = None):
        super().__init__(task=task, max_trials=max_trials)

    def call(self, x: Any) -> List[Dict]:
        y: Union[float, List[Dict]] = super().call(x)
        if isinstance(y, (float, np.ndarray)):
            y = float(y)
            # TODO: Should we check for a `name` attribute?
            task_name = getattr(self.unwrapped, "name", type(self.unwrapped).__name__.lower())
            results = [dict(name=task_name, type="objective", value=y)]
            return results
        return y


def array_to_trial(array: np.ndarray, space: Space) -> Trial:
    if array.ndim > 1:
        # remove the size-1 dims.
        array = np.squeeze(array)
    assert array.ndim == 1, "Assuming Points are 1-dimensional arrays for now."
    point_tuple: Tuple[Union[float, Tuple[float, ...]], ...] = regroup_dims(array, space=space)
    trial: Trial = tuple_to_trial(point_tuple, space=space)
    return trial


def trial_to_array(trial: Trial, space: Space) -> np.ndarray:
    # TODO: Not sure if `trial_to_tuple` reconstructs entries with the right shapes as well.
    point_tuple = trial_to_tuple(trial, space)
    flattened_point = flatten_dims(point_tuple, space)
    return np.array(flattened_point)


def params_to_array(params: Dict, space: Space) -> np.ndarray:
    trial = dict_to_trial(params, space=space)
    array = trial_to_array(trial, space=space)
    return array


class FixTaskDimensionsWrapper(TaskWrapper[TaskType]):
    """ Wrapper around a Task that fixes the values of some of its input dimensions.
    """
    def __init__(self, task: TaskType, fixed_dims: Dict[str, Any], max_trials: int = None):
        """ Wraps the provided task, fixing the dimensions in `fixed_dims`. """
        super().__init__(task=task, max_trials=max_trials)
        self.fixed_dims = fixed_dims
        # The whole space, from the wrapped task.
        self._full_space: Space = SpaceBuilder().build(self.task.get_search_space())
        # The part of the space that can be modified.
        self._space: Space = SpaceBuilder().build(self.get_search_space())

    def call(self, x: Any) -> List[Dict]:
        """ Calls the wrapped task, adding/changing some dimensions in `x` if necessary. """
        trial: Trial
        if isinstance(x, np.ndarray):
            trial = array_to_trial(x, space=self._full_space)
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
            return x

        if isinstance(x, np.ndarray):
            new_x = trial_to_array(trial, space=self._space)
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
