"""Wrapper that makes an algorithm warm-starteable by observing trials from different tasks.

The wrapper adds a `task_id` dimension to the search space, and uses the value of `task_id` to
differentiate between the current "target" experiment (`task_id==0`) and the other experiments
from the knowledge base (`task_id>0`).

The trials from the knowledge base that can fit within the same space are reused and given a
different value for a new "task_id" dimension.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger
from typing import Iterable

from orion.algo.space import Categorical, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoT
from orion.core.worker.algo_wrappers.transform_wrapper import (
    TransformWrapper,
    _copy_status_and_results,
)
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.knowledge_base import ExperimentInfo, KnowledgeBase
from orion.core.worker.warm_start.warm_starteable import WarmStarteable

logger = getLogger(__file__)


class MultiTaskWrapper(TransformWrapper[AlgoT], WarmStarteable):
    """Wrapper that makes the algo "multi-task" by adding a task id to the inputs."""

    def __init__(
        self,
        space: Space,
        algorithm: AlgoT,
    ):
        super().__init__(space=space, algorithm=algorithm)
        self.target_task_id: int = 0
        self.current_task_id: int = self.target_task_id

    def transform(self, trial: Trial) -> Trial:
        return self._add_task_id(trial, self.current_task_id)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self._remove_task_id(trial)

    # pylint: disable=arguments-differ
    @classmethod
    def transform_space(cls, space: Space, knowledge_base: KnowledgeBase) -> Space:
        """Transform the space, so that the algorithm that is passed to the constructor is already
        transformed.
        """
        # TODO: Should we have this dimension be larger than the current number of
        # experiments in the KB, in case some get added in the future?
        # TODO: Should the number of tasks here only count the experiments where the spaces are
        # compatible? (Currently only counts total number of experiments in the KB)
        # TODO: Could we use a Categorical over the experiment IDS instead of a Categorical with
        # ints? Would that help in some way?
        max_task_id = knowledge_base.n_stored_experiments + 1
        task_label_dimension = Categorical(
            "task_id",
            list(range(0, max_task_id)),
            default_value=0,
        )
        new_space = copy.deepcopy(space)
        if "task_id" in new_space:
            raise RuntimeError(f"Space already has a task_id dimension: {new_space}")
        new_space.register(task_label_dimension)
        return new_space

    def _is_compatible(self, trial_from_other_experiment: Trial) -> bool:
        """Used to check if a trial is compatible with the current experiment and can be reused.

        The trial will have a different task ID than the current task.

        NOTE: this assumes that the trial from the other experiment comes from the knowledge base,
        and that it therefore doesn't already have a task ID.
        """
        # TODO: Do we need to do something smarter here?
        return trial_from_other_experiment in self.space

    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentInfo, list[Trial]]]
    ) -> None:
        """Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsibility of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials:
            Dictionary mapping from ExperimentInfo objects (containing the experiment config) to
            the list of Trials associated with that experiment.
        """
        # Perform warm-starting using only the supported trials.
        logger.info(
            "Will warm-start using contextual information, since the algo isn't warm-starteable."
        )

        # Dict mapping from task ID to list of compatible trials.
        compatible_trials: dict[int, list[Trial]] = defaultdict(list)
        task_ids_to_experiments: dict[int, ExperimentInfo] = {}

        for i, (experiment_info, trials) in enumerate(warm_start_trials):
            # Start the task ids at 1, so the current experiment has task id 0.
            task_id = i + 1
            for trial in trials:
                # Drop the point if it doesn't fit inside the current space.
                # TODO: Add an optional 'translation' logic in the Knowledge Base implementation.
                if self._is_compatible(trial):
                    compatible_trials[task_id].append(trial)
                    task_ids_to_experiments[task_id] = experiment_info

            n_compatible_trials = len(compatible_trials[task_id])
            logger.info(
                f"Experiment {experiment_info} has {len(trials)} trials in "
                f"total, out of which {n_compatible_trials} were found to be "
                f"compatible with the target experiment."
            )

        if not compatible_trials:
            logger.info("No compatible trials detected.")
            return

        # Only keep trials that are new.
        new_compatible_trials = {
            task_id: [trial for trial in trials if trial not in self.algorithm.registry]
            for task_id, trials in compatible_trials.items()
        }

        if not new_compatible_trials:
            logger.info("No new new warm-starting trials detected.")
            return

        total_trials = sum(map(len, new_compatible_trials.values()))
        logger.info(
            "Algo will observe a total of %s trials from other experiments.",
            total_trials,
        )
        with self.algorithm.warm_start_mode():
            for task_id, trials in new_compatible_trials.items():
                logger.debug(
                    "Observing %s new trials from task %s", len(trials), task_id
                )

                with self.in_task(task_id):
                    # NOTE: self.observe adds the task ids to the trials (via the base class) that
                    # calls `self.transform`.
                    self.observe(trials)

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
    def _trials_from_target_task(self) -> Iterable[Trial]:
        return (
            trial
            for trial in self.algorithm.registry
            if get_task_id(trial) == self.target_task_id
        )

    @property
    def _trials_from_other_tasks(self) -> Iterable[Trial]:
        return (
            trial
            for trial in self.algorithm.registry
            if get_task_id(trial) != self.target_task_id
        )

    @property
    def is_done(self) -> bool:
        # TODO: Adjust this a bit, so the wrapped algo doesn't count the trials from other tasks in
        # its 'has_completed_max_trials'.
        # IDEA: Temporarily bump up the value of `max_trials` if set?

        total_trials = len(self.unwrapped.registry)
        trials_from_target_task = sum(
            get_task_id(trial) == self.target_task_id
            for trial in self.unwrapped.registry
        )
        trials_from_other_tasks = total_trials - trials_from_target_task
        logger.debug(
            f"Trials from target task: {trials_from_target_task}, "
            f"trials from other tasks: {trials_from_other_tasks} "
        )
        logger.debug(f"self.n_observed: {self.n_observed}")
        logger.debug(f"self.n_suggested: {self.n_suggested}")
        logger.debug(f"wrapped algo.is_done: {self.algorithm.is_done}")

        return (
            self.has_completed_max_trials
            or self.has_suggested_all_possible_values()
            # IDEA: Just ignore the `is_done` from the wrapped algo?
            or self.algorithm.is_done
        )

        # FIXME: Do the same logic as in the BaseAlgorithm.is_done, but only consider trials
        # from the target task.
        return super().is_done

    @property
    def has_completed_max_trials(self) -> bool:
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
        params = trial.params.copy()
        params["task_id"] = task_id
        trial_with_task_id = dict_to_trial(params, self.algorithm.space)
        trial_with_task_id = _copy_status_and_results(
            trial_with_status=trial, trial_with_params=trial_with_task_id
        )
        return trial_with_task_id

    def _remove_task_id(self, trial: Trial) -> Trial:
        params = trial.params.copy()
        _ = params.pop("task_id")
        trial_without_task_id = dict_to_trial(params, self.space)
        return trial_without_task_id


def get_task_id(trial: Trial) -> int:
    """Retrieves the task id of the given trial."""
    return trial.params["task_id"]
