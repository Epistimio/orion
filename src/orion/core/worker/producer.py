# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.producer` -- Produce and register samples to try
========================================================================

.. module:: producer
   :platform: Unix
   :synopsis: Suggest new parameter sets which optimize the objective.

"""
import logging

from orion.core.io.database import DuplicateKeyError
from orion.core.utils import format_trials

log = logging.getLogger(__name__)


class Producer(object):
    """Produce suggested sets of problem's parameter space to try out.

    It uses an `Experiment` object to poll for not yet observed trials which
    have been already evaluated and to register new suggestions (points of
    the parameter `Space`) to be evaluated.

    """

    def __init__(self, experiment, max_attempts=100):
        """Initialize a producer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        log.debug("Creating Producer object.")
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError("Experiment object provided to Producer has not yet completed"
                               " initialization.")
        self.algorithm = experiment.algorithms
        self.max_attempts = max_attempts

    @property
    def pool_size(self):
        """Pool-size of the experiment"""
        return self.experiment.pool_size

    def produce(self):
        """Create and register new trials."""
        sampled_points = 0
        n_attempts = 0

        while sampled_points < self.pool_size and n_attempts < self.max_attempts:
            n_attempts += 1
            log.debug("### Algorithm suggests new points.")

            new_points = self.algorithm.suggest(self.pool_size)

            for new_point in new_points:
                log.debug("#### Convert point to `Trial` object.")
                new_trial = format_trials.tuple_to_trial(new_point, self.space)
                try:
                    self.experiment.register_trial(new_trial)
                    log.debug("#### Register new trial to database: %s", new_trial)
                    sampled_points += 1
                except DuplicateKeyError:
                    log.debug("#### Duplicate sample. Updating algo to produce new ones.")
                    self.update()
                    break

        if n_attempts >= self.max_attempts:
            raise RuntimeError("Looks like the algorithm keeps suggesting trial configurations"
                               "that already exist in the database. Could be that you reached "
                               "a point of convergence and the algorithm cannot find anything "
                               "better. Or... something is broken. Try increasing `max_attempts` "
                               "or please report this error on "
                               "https://github.com/epistimio/orion/issues if something looks "
                               "wrong.")

    def update(self):
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
            self.experiment.update_history(completed_trials)
            self.algorithm.observe(points, results)
