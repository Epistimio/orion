# -*- coding: utf-8 -*-
"""
Produce and register samples to try
===================================

Suggest new parameter sets which optimize the objective.

"""
import logging

from orion.core.io.database import DuplicateKeyError
from orion.core.worker.trial import Trial

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
        self.params_hashes = set()
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
        new_points = self.algorithm.suggest(adjusted_pool_size)

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
                Trial.compute_trial_hash(
                    trial, ignore_experiment=True, ignore_lie=True, ignore_parent=True
                )
            )

    def update(self):
        """Pull all trials to update algorithm."""
        # TODO: Get rid of this inefficient pull when implementation shared algorithm state.
        trials = self.experiment.fetch_trials(with_evc_tree=True)
        self.num_trials = len(trials)
        self.num_broken = len([trial for trial in trials if trial.status == "broken"])

        self._update_algorithm(trials)

    def _update_algorithm(self, trials):
        """Pull trials to update local model."""
        log.debug("### Fetch trials to observe:")
        log.debug("### %s", trials)

        if trials:
            log.debug("### Observe them.")
            self.algorithm.observe(trials)
            self._update_params_hashes(trials)
