# -*- coding: utf-8 -*-
"""
Produce and register samples to try
===================================

Suggest new parameter sets which optimize the objective.

"""
import copy
import logging

from orion.core.io.database import DuplicateKeyError
from orion.core.worker.trial import Trial
from orion.core.worker.trials_history import TrialsHistory

log = logging.getLogger(__name__)


class Producer(object):
    """Produce suggested sets of problem's parameter space to try out.

    It uses an `Experiment` object to poll for not yet observed trials which
    have been already evaluated and to register new suggestions (points of
    the parameter `Space`) to be evaluated.

    """

    def __init__(self, experiment):
        """Initialize a producer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        log.debug("Creating Producer object.")
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError(
                "Experiment object provided to Producer has not yet completed"
                " initialization."
            )
        self.algorithm = experiment.algorithms
        self.strategy = experiment.producer["strategy"]
        self.naive_algorithm = None
        # TODO: Move trials_history into BaseAlgorithm during the refactoring of Algorithm with
        #       Strategist and Scheduler.
        self.trials_history = TrialsHistory()
        self.params_hashes = set()
        self.naive_trials_history = None
        self.num_trials = 0
        self.num_broken = 0

    def adjust_pool_size(self, pool_size):
        """Limit pool size if it would overshoot over max_trials"""
        num_pending = self.num_trials - self.num_broken
        num = max(self.experiment.max_trials - num_pending, 1)
        return min(num, pool_size)

    def produce(self, pool_size):
        """Create and register new trials."""
        adjusted_pool_size = self.adjust_pool_size(pool_size)
        log.debug(
            "### Algorithm attempts suggesting %s new points.", adjusted_pool_size
        )
        new_points = self.naive_algorithm.suggest(adjusted_pool_size)

        # Sync state of original algo so that state continues evolving.
        self.algorithm.set_state(self.naive_algorithm.state_dict)

        if not new_points and not self.algorithm.is_done:
            log.info(
                "Algo does not have more trials to sample."
                "Waiting for current trials to finish"
            )

        if not new_points:
            return 0

        return self.register_trials(new_points)

    def register_trials(self, new_points):
        """Register new sets of sampled parameters into the DB
        guaranteeing their uniqueness
        """
        registered_trials = 0
        for new_point in new_points:
            registered_trials += self.register_trial(new_point)

        return registered_trials

    def register_trial(self, new_trial):
        """Register a new set of sampled parameters into the DB
        guaranteeing their uniqueness

        Parameters
        ----------
        new_point: tuple
            tuple of values representing the hyperparameters values

        """
        # FIXME: Relying on DB to guarantee uniqueness
        # when the trial history will be held by that algo we can move that logic out of the DB
        try:
            self._prevalidate_trial(new_trial)
            new_trial.parents = self.naive_trials_history.children
            log.debug("#### Register new trial to database: %s", new_trial)
            self.experiment.register_trial(new_trial)
            self._update_params_hashes([new_trial])
            return 1

        except DuplicateKeyError:
            log.debug("#### Duplicate sample: %s", new_trial)
            return 0

    def _prevalidate_trial(self, new_trial):
        """Verify if trial is not in parent history"""
        if (
            Trial.compute_trial_hash(new_trial, ignore_experiment=True)
            in self.params_hashes
        ):
            raise DuplicateKeyError

    def _update_params_hashes(self, trials):
        """Register locally all param hashes of trials"""
        for trial in trials:
            self.params_hashes.add(
                Trial.compute_trial_hash(trial, ignore_experiment=True, ignore_lie=True)
            )

    def update(self):
        """Pull all trials to update model with completed ones and naive model with non completed
        ones.
        """
        trials = self.experiment.fetch_trials(with_evc_tree=True)
        self.num_trials = len(trials)
        self.num_broken = len([trial for trial in trials if trial.status == "broken"])

        self._update_algorithm(
            [trial for trial in trials if trial.status == "completed"]
        )
        self._update_naive_algorithm(
            [trial for trial in trials if trial.status != "completed"]
        )

    def _update_algorithm(self, completed_trials):
        """Pull newest completed trials to update local model."""
        log.debug("### Fetch completed trials to observe:")

        new_completed_trials = []
        for trial in completed_trials:
            # if trial not in self.trials_history:
            if not self.algorithm.has_observed(trial):
                new_completed_trials.append(trial)

        log.debug("### %s", new_completed_trials)

        if new_completed_trials:
            log.debug("### Observe them.")
            self.trials_history.update(new_completed_trials)
            self.algorithm.observe(new_completed_trials)
            self.strategy.observe(new_completed_trials)
            self._update_params_hashes(new_completed_trials)

    def _produce_lies(self, incomplete_trials):
        """Add fake objective results to incomplete trials

        Then register the trials in the db
        """
        log.debug("### Fetch active trials to observe:")
        lying_trials = []
        log.debug("### %s", incomplete_trials)

        for trial in incomplete_trials:
            log.debug("### Use defined ParallelStrategy to assign them fake results.")
            lying_result = self.strategy.lie(trial)
            if lying_result is not None:
                lying_trial = copy.deepcopy(trial)
                lying_trial.results.append(lying_result)
                lying_trials.append(lying_trial)
                log.debug("### Register lie to database: %s", lying_trial)
                lying_trial.parents = self.trials_history.children
                try:
                    self.experiment.register_lie(lying_trial)
                except DuplicateKeyError:
                    log.debug(
                        "#### Duplicate lie. No need to register a duplicate in DB."
                    )

        return lying_trials

    def _update_naive_algorithm(self, incomplete_trials):
        """Pull all non completed trials to update naive model."""
        self.naive_algorithm = copy.deepcopy(self.algorithm)
        self.naive_trials_history = copy.deepcopy(self.trials_history)
        log.debug("### Create fake trials to observe:")
        lying_trials = self._produce_lies(incomplete_trials)
        log.debug("### %s", lying_trials)
        if lying_trials:
            log.debug("### Observe them.")
            self.naive_trials_history.update(lying_trials)
            self.naive_algorithm.observe(lying_trials)
            self._update_params_hashes(lying_trials)
