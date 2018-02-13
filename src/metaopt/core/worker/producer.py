# -*- coding: utf-8 -*-
"""
:mod:`metaopt.core.worker.producer` -- Produce and register samples to try
==========================================================================

.. module:: producer
   :platform: Unix
   :synopsis: Suggest new parameter sets which optimize the objective.

"""
import logging

import six

from metaopt.core.utils import (format_trials, SingletonType)

log = logging.getLogger(__name__)


@six.add_metaclass(SingletonType)
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

    def produce(self):
        """Create and register new trials."""
        log.debug("### Suggest new ones.")
        new_points = self.algorithm.suggest(self.num_new_trials)

        log.debug("### Convert them to `Trial` objects.")
        new_trials = list(map(lambda data: format_trials.tuple_to_trial(data, self.space),
                              new_points))

        log.debug("### Register to database: %s", new_trials)
        self.experiment.register_trials(new_trials)

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
            self.algorithm.observe(points, results)
