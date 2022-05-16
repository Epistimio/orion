"""Wrapper that makes an algorithm multi-task and feeds it data from the knowledge base.

"""
from __future__ import annotations

import copy
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger
from typing import Iterable, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Categorical, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.worker.algo_wrapper import AlgoWrapper
from orion.core.worker.knowledge_base import AbstractKnowledgeBase, ExperimentInfo
from orion.core.worker.primary_algo import _copy_status_and_results
from orion.core.worker.trial import Trial
from orion.core.worker.warm_starting.warm_starteable import is_warmstarteable

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)
log = getLogger(__file__)


def create_algo(
    algo_type: type[AlgoType],
    space: Space,
    knowledge_base: AbstractKnowledgeBase | None = None,
    **algo_kwargs,
) -> AlgoType | MultiTaskAlgo[AlgoType]:
    """Creates an algorithm of the given type, wrapping it with a MultiTaskAlgo wrapper if it isn't
    warm-startable and if there are experiments in the knowledge base.
    """

    if (
        knowledge_base is None
        or is_warmstarteable(algo_type)
        or knowledge_base.n_stored_experiments == 0
    ):
        log.debug("Not using a multi-task wrapper.")
        return algo_type(space, **algo_kwargs)

    log.debug(
        "Chosen algorithm is NOT warm-starteable, enabling this multi-task wrapper."
    )
    # TODO: Should we have this dimension be larger than the current number of
    # experiments in the KB, in case some get added in the future?
    max_task_id = knowledge_base.n_stored_experiments + 1
    task_label_dimension = Categorical(
        "task_id",
        list(range(0, max_task_id)),
        default_value=0,
    )
    space_without_task_ids = copy.deepcopy(space)
    space_with_task_ids = copy.deepcopy(space)
    space_with_task_ids.register(task_label_dimension)

    algorithm = algo_type(space=space_with_task_ids, **algo_kwargs)
    return MultiTaskAlgo(
        space=space_without_task_ids, algorithm=algorithm, knowledge_base=knowledge_base
    )


class MultiTaskAlgo(AlgoWrapper[AlgoType]):
    """Wrapper that makes the algo "multi-task" by adding a task id to the inputs."""

    def __init__(
        self,
        space: Space,
        algorithm: AlgoType,
        knowledge_base: AbstractKnowledgeBase,
    ):
        log.debug(
            f"Creating a MultiTaskAlgo wrapper: space {space}, algo config: {algorithm.configuration}"
        )
        self.knowledge_base = knowledge_base
        self.current_task_id: int = 0

        # TODO: Use `self.registry` only for the current (target) task.
        # Let `self.algorithm.registry` contain trials from all tasks, with task IDs added,
        # including the target task.

    def transform(self, trial: Trial) -> Trial:
        return self._add_task_id(trial, self.current_task_id)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self._remove_task_id(trial)

    def _is_compatible(self, trial_from_other_experiment: Trial) -> bool:
        """Used to check if a trial is compatible with the current experiment and can be reused for warm-starting.
        The trial will have a different task ID than the current task.

        NOTE: this assumes that the trial from the other experiment comes from the knowledge base,
        and that is therefore doesn't already have a task ID.
        """
        # TODO: Do we need to do something smarter here?
        return trial_from_other_experiment in self.space

    def warm_start(self, warm_start_trials: dict[ExperimentInfo, list[Trial]]) -> None:
        """Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsibility of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials : Dict[ExperimentInfo, List[Trial]]
            Dictionary mapping from ExperimentInfo objects (containing the experiment config) to
            the list of Trials associated with that experiment.
        """
        # Perform warm-starting using only the supported trials.
        log.info(
            "Will warm-start using contextual information, since the algo isn't warm-starteable."
        )

        # Dict mapping from task ID to list of compatible trials.
        compatible_trials: dict[int, list[Trial]] = defaultdict(list)
        task_ids_to_experiments: dict[int, ExperimentInfo] = {}

        for i, (experiment_info, trials) in enumerate(warm_start_trials.items()):
            # Start the task ids at 1, so the current experiment has task id 0.
            task_id = i + 1
            for trial in trials:
                # Drop the point if it doesn't fit inside the current space.
                # TODO: Do we want to 'translate' the point in this case?
                if self._is_compatible(trial):
                    compatible_trials[task_id].append(trial)
                    task_ids_to_experiments[task_id] = experiment_info

            n_compatible_trials = len(compatible_trials[task_id])
            log.info(
                f"Experiment {experiment_info} has {len(trials)} trials in "
                f"total, out of which {n_compatible_trials} were found to be "
                f"compatible with the target experiment."
            )

        if not compatible_trials:
            log.info("No compatible trials detected.")
            return

        # Only keep trials that are new.
        new_compatible_trials = {
            task_id: [trial for trial in trials if trial not in self.algorithm.registry]
            for task_id, trials in compatible_trials.items()
        }

        if not new_compatible_trials:
            log.info("No new new warm-starting trials detected.")
            return

        new_trials_with_task_ids = {
            task_id: [self._add_task_id(trial, task_id) for trial in trials]
            for task_id, trials in new_compatible_trials.items()
        }

        with self.algorithm.warm_start_mode():
            total_new_points = sum(map(len, new_trials_with_task_ids.values()))
            log.info(f"About to observe {total_new_points} new warm-starting points!")
            self.algorithm.observe(new_trials_with_task_ids)

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm **in the target task**"""
        # TODO: Unsure about this.
        return super().n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm"""
        # TODO: Unsure about this.
        return super().n_observed

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)

    @property
    def trials_from_target_task(self) -> Iterable[Trial]:
        return (trial for trial in self.algorithm.registry if get_task_id(trial) == 0)

    @property
    def trials_from_other_tasks(self) -> Iterable[Trial]:
        return (trial for trial in self.algorithm.registry if get_task_id(trial) != 0)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        # NOTE: DEBUGGING:
        total_trials = len(self.unwrapped.registry)
        trials_from_target_task = sum(
            get_task_id(trial) == 0 for trial in self.unwrapped.registry
        )
        trials_from_other_tasks = total_trials - trials_from_target_task
        log.debug(
            f"Trials from target task: {trials_from_target_task}, "
            f"trials from other tasks: {trials_from_other_tasks} "
        )
        log.debug(f"self.n_observed: {self.n_observed}")
        log.debug(f"self.n_observed: {self.n_suggested}")
        log.debug(f"wrapped algo.is_done: {self.algorithm.is_done}")

        # FIXME: Do the same logic as in the BaseAlgorithm.is_done, but only consider trials
        # from the target task.
        if trials_from_target_task >= self.space.cardinality:
            return True

        max_trials = getattr(self, "max_trials", float("inf"))
        if trials_from_target_task >= max_trials:
            return True

        return False

    @contextmanager
    def in_task(self, task_id: int):
        """Contextmanager that temporarily changes the value of `self.current_task_id`
        to `task_id` and restores the original value after exiting the with block.
        """
        previous_task_id = self.current_task_id
        self.current_task_id = task_id
        yield
        self.current_task_id = previous_task_id

    def _add_task_id(self, trial: Trial, task_id: int) -> Trial:
        # NOTE: Need to do a bit of gymnastics here because the points are expected
        # to be tuples. In order to add the task id at the right index, we convert
        # them like so:
        # Trial -> dict (.params) -> dict (add task_id) -> Trial.
        params = trial.params.copy()
        params["task_id"] = task_id

        # TODO: Should we copy over the status and everything else?
        trial_with_task_id = dict_to_trial(params, self.algorithm.space)

        trial_with_task_id = _copy_status_and_results(trial, trial_with_task_id)
        return trial_with_task_id

    def _remove_task_id(self, trial: Trial) -> Trial:
        params = trial.params.copy()
        _ = params.pop("task_id")
        trial_without_task_id = dict_to_trial(params, self.space)
        return trial_without_task_id


def get_task_id(trial: Trial) -> int:
    return trial.params.get("task_id", 0)
