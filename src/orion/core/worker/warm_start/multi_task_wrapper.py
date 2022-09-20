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

from orion.algo.space import Categorical, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoT
from orion.core.worker.algo_wrappers.transform_wrapper import (
    TransformWrapper,
    _copy_status_and_results,
)
from orion.core.worker.experiment_config import ExperimentConfig
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start.knowledge_base import KnowledgeBase
from orion.core.worker.warm_start.warm_starteable import WarmStarteable

logger = getLogger(__file__)

TARGET_TASK_ID = 0

# pylint: disable=too-many-public-methods
class MultiTaskWrapper(TransformWrapper[AlgoT], WarmStarteable):
    """Wrapper that makes the algo "multi-task" by adding a task id to the inputs."""

    def __init__(
        self,
        space: Space,
        algorithm: AlgoT,
    ):
        super().__init__(space=space, algorithm=algorithm)
        self.current_task_id: int = TARGET_TASK_ID
        self._total_warm_start_trials = 0
        self._max_trials: int | None = None

    def transform(self, trial: Trial) -> Trial:
        return self._add_task_id(trial, self.current_task_id)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self._remove_task_id(trial)

    # pylint: disable=arguments-differ
    @classmethod
    def transform_space(cls, space: Space, knowledge_base: KnowledgeBase) -> Space:
        """Transform an (outer) space, returning the (inner) space of the wrapped algorithm.

        This returns what will become `self.algorithm.space` in an instance of this class.
        """
        if "task_id" in space:
            raise RuntimeError(f"Space already has a task_id dimension: {space}")
        # NOTE: The task dimension here has the size of the total number of experiments in the KB,
        # not the number of compatible experiments.
        # This is because we don't yet know how many of them are directly compatible.
        # Note: The +1 here represents the current experiment.
        max_task_id = 1 + knowledge_base.n_stored_experiments
        # NOTE: We set a non-zero probability only on the task_id==target dimension, since:
        # 1. It also doesn't really make sense for the algo to try to optimize the other tasks, and
        # 2. this makes the algorithm only suggest trials with task_id==target, to reduce the risk
        #    of collisions if it were to suggest the same trials with different task ids.
        task_label_dimension = Categorical(
            "task_id",
            {
                task_id: (1 if task_id == TARGET_TASK_ID else 0)
                for task_id in range(0, max_task_id)
            },
            default_value=TARGET_TASK_ID,
        )
        new_space = copy.deepcopy(space)
        new_space.register(task_label_dimension)
        return new_space

    def _is_compatible(self, trial_from_other_experiment: Trial) -> bool:
        """Used to check if a trial is compatible with the current experiment and can be reused.

        The trial will have a different task ID than the current task.

        NOTE: this assumes that the trial from the other experiment comes from the knowledge base,
        and that it therefore doesn't already have a task ID.
        """
        return trial_from_other_experiment in self.space

    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentConfig, list[Trial]]]
    ) -> None:
        """Observe some of the given trials to warm-start the HPO algorithm.

        These experiments and their trials were fetched from some storage, and
        are believed to be somewhat similar to the current on-going experiment.
        By default, this wrapper only considers the trials which are compatible
        with the current space.

        Parameters
        ----------
        warm_start_trials:
            Dictionary mapping from ExperimentConfig objects (containing the experiment config) to
            the list of Trials associated with that experiment.
        """
        logger.info(
            "Warm-starting using only multi-task training with compatible trials, since the algo "
            "isn't warm-starteable."
        )

        # Dict mapping from task ID to list of compatible trials.
        compatible_trials: dict[int, list[Trial]] = defaultdict(list)
        task_ids_to_experiments: dict[int, ExperimentConfig] = {}
        for i, (experiment_info, trials) in enumerate(warm_start_trials):
            # Start the task ids at 1, so the current experiment has task id 0.
            task_id = i + 1
            for trial in trials:
                # Drop the point if it doesn't fit inside the current space.
                # IDEA: Possibly add some 'translation' logic in an eventual Knowledge Base
                # implementation.
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
            logger.info("No compatible trials found!")
            return

        # Note: If there are no compatible trials found, this would allow multiple attempts at
        # warm-starting. This isn't used though.
        if self._total_warm_start_trials:
            raise RuntimeError("The algorithm can only be warm-started once!")
        self._total_warm_start_trials = sum(map(len, compatible_trials.values()))

        logger.info(
            "Algo will observe a total of %s trials from other experiments.",
            self._total_warm_start_trials,
        )

        for task_id, trials in compatible_trials.items():
            logger.debug("Observing %s new trials from task %s", len(trials), task_id)
            assert task_id != TARGET_TASK_ID
            with self.in_task(task_id):
                # NOTE: self.observe saves those trials in our registry, and adds the task ids
                # to the trials and passes that calls `self.transform`.
                # This also reuses the collision handling logic from the Transform wrapper.
                self.observe(trials)

        if self._max_trials is not None:
            wrapped_algo_max_trials = self._max_trials + self._total_warm_start_trials
            logger.debug(
                "Setting the max_trials of the wrapped algo to %s rather than %s, to account for "
                "the trials observed during warm-starting.",
                wrapped_algo_max_trials,
                self._max_trials,
            )
            self.algorithm.max_trials = wrapped_algo_max_trials

    def observe(self, trials: list[Trial]) -> None:
        # NOTE: This uses the same collision-handling logic as the Transform wrapper.
        return super().observe(trials)

    def register(self, trial: Trial) -> None:
        if self.current_task_id != TARGET_TASK_ID:
            # Don't register it. We're in warm-start mode, and this trial comes from a different
            # task. We observe the trials from other tasks using `self.observe`, rather than
            # self.algorithm.observe, since we want to reuse the collision-handling logic of the
            # TransformWrapper.

            # TODO: There's maybe a slight problem with this: The TransformWrapper base-class will
            # registers trials in the registry mapping, and the registry mapping usually assumes
            # that the source and target registries contain the source and target trials. However,
            # here, we don't actually register the source trial in our source registry.
            # This isn't currently causing any issues, and I'm currently unable to imagine a
            # scenario in which it would cause problems, although it might be possible.
            # Leaving this as a TODO for now.
            return
        super().register(trial)

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm, excluding warm-starting."""
        return super().n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm, excluding warm-starting."""
        return super().n_observed

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)

    @property
    def max_trials(self) -> int | None:
        """Maximum number of trials to run, or `None` when there is no limit."""
        return self._max_trials

    @max_trials.setter
    def max_trials(self, value: int | None) -> None:
        # NOTE: We delay setting the max_trials property until we know how many trials we will pass
        # to the algo during warm-starting.
        self._max_trials = value

    @property
    def is_done(self) -> bool:
        # NOTE: Uses the same logic as in BaseAlgorithm.is_done, but only consider trials
        # from the target task.
        return super().is_done

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
            trial_with_status=trial, trial=trial_with_task_id
        )
        return trial_with_task_id

    def _remove_task_id(self, trial: Trial) -> Trial:
        params = trial.params.copy()
        _ = params.pop("task_id")
        trial_without_task_id = dict_to_trial(params, self.space)
        trial_without_task_id = _copy_status_and_results(
            trial_with_status=trial, trial=trial_without_task_id
        )
        return trial_without_task_id
