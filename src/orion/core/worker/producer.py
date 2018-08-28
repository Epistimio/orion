# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.producer` -- Produce and register samples to try
========================================================================

.. module:: producer
   :platform: Unix
   :synopsis: Suggest new parameter sets which optimize the objective.

"""
import copy
import logging

from orion.core.utils import format_trials

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
        self.num_new_trials = experiment.pool_size
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError("Experiment object provided to Producer has not yet completed"
                               " initialization.")
        self.algorithm = experiment.algorithms
        self.strategy = experiment.producer['strategy']
        self.naive_algorithm = None

    def produce(self):
        """Create and register new trials."""
        log.debug("### Suggest new ones.")
        for ith_trial in range(self.num_new_trials):
            new_point = self.naive_algorithm.suggest(1)[0]

            log.debug("### Convert %d-th to `Trial` objects.", ith_trial)
            new_trial = format_trials.tuple_to_trial(new_point, self.space)

            log.debug("### Register to database: %s", new_trial)
            self.experiment.register_trial(new_trial)

    def update(self):
        """Pull newest completed trials and all non completed trials to update naive model."""
        self._update_algorithm()
        self._update_naive_algorithm()

    def _update_algorithm(self):
        """Pull newest completed trials to update local model."""
        log.debug("### Fetch trials to observe:")
        completed_trials = self.experiment.fetch_completed_trials()
        log.debug("### %s", completed_trials)

        if completed_trials:
            log.debug("### Convert them to list of points and their results.")
            points = list(map(lambda trial: format_trials.trial_to_tuple(trial, self.space),
                              completed_trials))
            results = list(map(format_trials.get_trial_results, completed_trials))

            log.debug("### Observe them.")
            self.algorithm.observe(points, results)
            self.strategy.observe(points, results)

    def _produces_lies(self):
        """Add fake objective results to incomplete trials

        Then register the trials in the db
        """
        log.debug("### Fetch active trials to observe:")
        incomplete_trials = self.experiment.fetch_active_trials()
        log.debug("### %s", incomplete_trials)

        for trial in incomplete_trials:
            log.debug("### Use defined ParallelStrategy to assign them fake results.")
            lying_result = self.strategy.lie(trial)
            if lying_result is not None:
                trial.results.append(lying_result)
            log.debug("### Register lie to database: %s", trial)
            self.experiment.register_trial(trial)

        return incomplete_trials

    def _update_naive_algorithm(self):
        """Pull all non completed trials to update naive model."""
        self.naive_algorithm = copy.deepcopy(self.algorithm)
        log.debug("### Create fake trials to observe:")
        lying_trials = self._produces_lies()
        log.debug("### %s", lying_trials)
        if lying_trials:
            log.debug("### Convert them to list of points and their results.")
            points = list(map(lambda trial: format_trials.trial_to_tuple(trial, self.space),
                              lying_trials))
            results = list(map(format_trials.get_trial_results, lying_trials))

            log.debug("### Observe them.")
            self.naive_algorithm.observe(points, results)
