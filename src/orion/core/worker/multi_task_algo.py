"""Wrapper that makes an algorithm multi-task and feeds it data from the knowledge base.

"""
import inspect
import textwrap
import warnings
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy as np
from orion.algo.base import BaseAlgorithm, infer_trial_id
from orion.algo.space import Categorical, Space
from orion.client import ExperimentClient
from orion.core.utils.format_trials import trial_to_tuple, tuple_to_trial, dict_to_trial
from orion.core.worker.experiment import Experiment
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.trial import Trial
from orion.storage.base import Storage

try:
    from functools import singledispatchmethod
except ImportError:
    from singledispatchmethod import singledispatchmethod

from .algo_wrapper import AlgoWrapper, Point


log = getLogger(__file__)

from .knowledge_base import AbstractKnowledgeBase


class WarmStarteable(ABC):
    """ Base class for Algorithms which can leverage 'related' past trials to bootstrap
    their optimization process.
    """

    @abstractmethod
    def warm_start(self, warm_start_trials: Dict["ExperimentInfo", List[Trial]]):
        """ Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsability of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials : Dict[Mapping, List[Trial]]
            Dictionary mapping from ExperimentInfo objects (dataclasses containing the
            experiment config) to the list of Trials associated with that experiment.
        """
        raise NotImplementedError(
            f"Algorithm of type {type(self)} isn't warm-starteable yet."
        )

    # @contextmanager
    # @abstractmethod
    # def warm_start_mode(self):
    #     pass


def is_warm_starteable(
    algo_or_config: Union[BaseAlgorithm, Type[BaseAlgorithm], Dict[str, Any]]
) -> bool:
    """Returns wether the given algo, algo type, or algo config, supports warm-starting.

    Parameters
    ----------
    algo_or_config : Union[BaseAlgorithm, Type[BaseAlgorithm], Dict[str, Any]]
        Algorithm instance, type of algorithm, or algorithm configuration dictionary.

    Returns
    -------
    bool
        Wether the input is or describes a warm-starteable algorithm.
    """
    available_algos = {
        c.__name__.lower(): c for c in list(BaseAlgorithm.__subclasses__())
    }
    if inspect.isclass(algo_or_config):
        return issubclass(algo_or_config, WarmStarteable)
    if isinstance(algo_or_config, str):
        algo_type = available_algos.get(algo_or_config.lower())
        return algo_type is not None and issubclass(algo_type, WarmStarteable)
    if isinstance(algo_or_config, dict):
        first_key = list(algo_or_config)[0]
        return len(algo_or_config) == 1 and is_warm_starteable(first_key)
    return isinstance(
        getattr(algo_or_config, "unwrapped", algo_or_config), WarmStarteable
    )


class MultiTaskAlgo(AlgoWrapper):
    """Wrapper that makes the algo "multi-task" by concatenating the task ids to the inputs.
    """

    def __init__(
        self,
        space: Space,
        algorithm_config: Dict,
        knowledge_base: AbstractKnowledgeBase,
    ):
        log.info(
            f"Creating a MultiTaskAlgo wrapper: space {space}, algo config: {algorithm_config}"
        )
        self.knowledge_base = knowledge_base
        self._enabled: bool
        if is_warm_starteable(algorithm_config):
            log.info(
                "Chosen algorithm is warm-starteable, disabling this multi-task wrapper."
            )
            self.algorithm = PrimaryAlgo(space=space, algorithm_config=algorithm_config)
            self.space = self.algorithm.space
            self._enabled = False
            return  # Return, just to indicate we don't do the other modifications below.
        else:
            log.info(
                "Chosen algorithm is NOT warm-starteable, enabling this multi-task wrapper."
            )
            self._enabled = True

        # Figure out the number of potential tasks using the KB.
        # TODO: The current Experiment might already be registered in the KB?
        if knowledge_base.n_stored_experiments < 1:
            # TODO: Should probably turn the 'warm-start' portion of this 'wrapper' off,
            # because it will probably reduce the effectiveness of the algo to have to
            # optimize with an uninformative 'task_id' dimension.
            warnings.warn(
                RuntimeWarning(
                    f"Making the Algo 'multi-task', even though there are only"
                    f"{knowledge_base.n_stored_experiments} experiments in the "
                    f"Knowledge base. This might reduce the efficiency of the algo."
                )
            )
        # TODO: Should we have this dimension be larger than the current number of
        # experiments in the KB, in case some get added in the future?
        max_task_id = self.knowledge_base.n_stored_experiments + 1
        task_label_dimension = Categorical(
            "task_id", list(range(0, max_task_id)), default_value=0,
        )
        space_without_task_ids = Space(space.copy())
        space_with_task_ids = Space(space.copy())
        space_with_task_ids.register(task_label_dimension)
        assert "task_id" in space_with_task_ids
        assert "task_id" not in space_without_task_ids

        self.algorithm = PrimaryAlgo(
            space=space_with_task_ids, algorithm_config=algorithm_config
        )
        self._space = space_without_task_ids
        self.current_task_id = 0
        assert "task_id" not in self._space
        assert "task_id" not in self.space
        # TODO: For some optimizers this exact key won't be found, there is a View(OneHotEncode(...))
        # assert "task_id" in self.algorithm.transformed_space, self.algorithm.transformed_space
        # assert "task_id" in self.algorithm.space
        self.transformed_space = self.space

    def suggest(self, num: int = 1) -> List[Point]:
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = self.algorithm.suggest(num=num)
        if not self._enabled:
            return points

        # Remove the task ids from these points:
        # NOTE: These predicted task ids aren't currently used.

        points_without_task_ids = [self._remove_task_id(point) for point in points]
        for point in points_without_task_ids:
            if point not in self.space:
                raise ValueError(
                    textwrap.dedent(
                        f"""\
                        Point is not contained in space:
                        Point: {point}
                        Space: {self.space}
                        """
                    )
                )
        return points_without_task_ids

    def observe(
        self, points: List[Point], results: Union[List[float], List[Dict]]
    ) -> None:
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        # Add the task ids to the points, because we assume that the points passed to
        # `observe` are from the current task.
        if self._enabled:
            # Add the task ids to the points.
            points = [
                self._add_task_id(point, self.current_task_id) for point in points
            ]
        return self.algorithm.observe(points, results)

    def warm_start(self, warm_start_trials: Dict[Mapping, List[Trial]]) -> None:
        """ Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsability of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials : Dict[Mapping, List[Trial]]
            Dictionary mapping from ExperimentInfo objects (dataclasses containing the
            experiment config) to the list of Trials associated with that experiment.
        """
        # Being asked to warm-start the algorithm.
        if is_warm_starteable(self.algorithm):
            log.debug(f"Warm starting the algo with trials {warm_start_trials}")
            self.algorithm.warm_start(warm_start_trials)
            log.info("Algorithm was successfully warm-started, returning.")
            return

        # Perform warm-starting using only the supported trials.
        log.info(
            "Will warm-start using contextual information, since the algo "
            "isn't warm-starteable."
        )

        compatible_trials: List[Trial] = []
        compatible_points: List[Tuple] = []
        task_ids: list[int] = []

        for i, (experiment_info, trials) in enumerate(warm_start_trials.items()):
            # exp_space = experiment_info.space
            # from warmstart.utils.api_config import hparam_class_from_orion_space_dict
            # exp_hparam_class = hparam_class_from_orion_space_dict(exp_space)
            log.debug(f"Experiment {experiment_info.name} has {len(trials)} trials.")
            n_incompatible_trials = 0

            for trial in trials:
                try:
                    # TODO: Use something smarter than `trial_to_tuple`, since it
                    # doesn't adjust the points if it's just missing a default value.
                    point = trial_to_tuple(trial=trial, space=self.space)
                    # Drop the point if it doesn't fit inside the current space.
                    # TODO: Do we want to 'translate' the point in this case?
                    # if self.translate_points:
                    #     point = self.translator.translate(point)
                    if point not in self.space:
                        raise ValueError
                    # Add the task id to the point:
                    # The task ID of the given experiment is `i + 1`, so that the task
                    # id 0 is reserved for the trials of the current task.
                    # TODO: This assumes that the trials from the knowledge base don't
                    # include those of the current experiment.
                    task_id = i + 1

                    # point = self._add_task_id(point, task_id)
                    # trial = self._add_task_id(trial, task_id)

                    compatible_trials.append(trial)
                    compatible_points.append(point)
                    task_ids.append(task_id)

                except ValueError as e:
                    log.error(f"Can't reuse trial {trial}: {e}")
                    n_incompatible_trials += 1

            total_trials = len(trials)
            n_compatible_trials = total_trials - n_incompatible_trials
            log.info(
                f"Experiment {experiment_info.name} has {len(trials)} trials in "
                f"total, out of which {n_compatible_trials} were found to be "
                f"compatible with the target experiment."
            )

        if not compatible_trials:
            log.info("No compatible trials detected.")
            return

        # Only keep trials that are new.
        # TODO: Check that the task dim doesnt interfere with this:
        new_trials, new_points, new_task_ids = list(
            zip(
                *[
                    (trial, point, task_id)
                    for trial, point, task_id in zip(
                        compatible_trials, compatible_points, task_ids
                    )
                    if infer_trial_id(point) not in self.unwrapped._warm_start_trials
                ]
            )
        )
        if not new_trials:
            log.info("No new new warm-starting trials detected.")
            return

        with self.algorithm.warm_start_mode():
            new_points_with_task_ids = [
                self._add_task_id(point, task_id)
                for point, task_id in zip(new_points, new_task_ids)
            ]
            new_trials_with_task_ids = [
                self._add_task_id(trial, task_id)
                for trial, task_id in zip(new_trials, new_task_ids)
            ]
            # for new_trial, new_trial_with_task_id in zip(new_trials, new_trials_with_task_ids):
            #     assert False, (new_trial, new_trial_with_task_id)

            log.info(f"About to observe {len(new_points)} new warm-starting points!")
            # TODO: Not quite sure I understand what the 'results' is supposed to be!
            # results = [trial.objective for trial in new_trials]
            results = [
                {"objective": trial.objective.value, "constraint": [], "gradient": None}
                for trial in new_trials_with_task_ids
            ]
            self.algorithm.observe(new_points_with_task_ids, results)

    def score(self, point: Union[Tuple, np.ndarray]) -> float:
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score for any parameter (no preference).
        """
        assert point in self.space
        if self._enabled:
            point = self._add_task_id(point, self.current_task_id)
        return self.algorithm.score(point)

    def judge(self, point: Point, measurements: Any) -> Optional[Dict]:
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        assert point in self._space
        if self._enabled:
            point = self._add_task_id(point, self.current_task_id)
        return self.algorithm.judge(
            self.transformed_space.transform(point), measurements
        )

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if not self._enabled:
            return self.algorithm.is_done
        # NOTE: DEBUGGING:
        total_trials = len(self.unwrapped._trials_info)
        trials_from_target_task = sum(
            self.get_task_id(point) == 0
            for point, result in self.unwrapped._trials_info.values()
        )
        trials_from_other_tasks = total_trials - trials_from_target_task
        log.info(
            f"Trials from target task: {trials_from_target_task}, trials from other tasks: {trials_from_other_tasks}"
        )

        if trials_from_target_task >= self.space.cardinality:
            return True

        if trials_from_target_task >= getattr(self, "max_trials", float("inf")):
            return True

        return False

    @singledispatchmethod
    def _add_task_id(self, point: Any, task_id: int) -> Any:
        raise NotImplementedError(point)

    @_add_task_id.register(np.ndarray)
    @_add_task_id.register(tuple)
    def _add_task_id_to_point(self, point: Any, task_id: int) -> Tuple[int, Any]:
        # NOTE: Need to do a bit of gymnastics here because the points are expected
        # to be tuples. In order to add the task id at the right index, we convert
        # them like so:
        # Tuple -> Trial -> dict (.params) -> dict (add task_id) -> Trial -> Tuple.
        trial = tuple_to_trial(point, self.space)
        trial_with_task_id = self._add_task_id(trial, task_id)
        point_with_task_id = trial_to_tuple(trial_with_task_id, self.algorithm.space)
        return point_with_task_id

    @_add_task_id.register
    def _add_task_id_to_trial(self, trial: Trial, task_id: int) -> Trial:
        # NOTE: Need to do a bit of gymnastics here because the points are expected
        # to be tuples. In order to add the task id at the right index, we convert
        # them like so:
        # Trial -> dict (.params) -> dict (add task_id) -> Trial.
        params = trial.params.copy()
        params["task_id"] = task_id
        # TODO: Should we copy over the status and everything else?
        trial_with_task_id = dict_to_trial(params, self.algorithm.space)
        for attribute in trial.__slots__:
            value = getattr(trial, attribute)
            if "params" not in attribute:
                setattr(trial_with_task_id, attribute, value)
        return trial_with_task_id

    @singledispatchmethod
    def _remove_task_id(self, point: Any) -> Any:
        raise NotImplementedError(point)

    @_remove_task_id.register(tuple)
    def _remove_task_id_from_point(self, point: Tuple) -> Tuple:
        trial = tuple_to_trial(point, self.algorithm.space)
        trial_without_task_id = self._remove_task_id(trial)
        point_without_task_id = trial_to_tuple(trial_without_task_id, self.space)
        return point_without_task_id

    @_remove_task_id.register(Trial)
    def _remove_task_id_from_trial(self, trial: Trial) -> Trial:
        params = trial.params.copy()
        _ = params.pop("task_id")
        trial_without_task_id = dict_to_trial(params, self.space)
        return trial_without_task_id

    @singledispatchmethod
    def get_task_id(self, point: Any) -> int:
        raise NotImplementedError(point)

    @get_task_id.register
    def get_task_id_from_trial(self, trial: Trial) -> int:
        return trial.params["task_id"] != 0

    @get_task_id.register(tuple)
    @get_task_id.register(np.ndarray)
    def get_task_id_from_point(self, point: Union[tuple, np.ndarray]) -> int:
        # TODO: Bug in tuple_to_trial? I'm getting point = (0, 1.23, 4.56) instead of
        # (0, [1.23, 4.56])
        if isinstance(point, tuple):
            if len(point) != len(self.algorithm.space):
                task_id, *point_without_task_id = point
                assert isinstance(task_id, int)
                return task_id
        trial = tuple_to_trial(point, self.algorithm.space)
        return self.get_task_id(trial)
