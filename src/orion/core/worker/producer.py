# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.producer` -- Produce and register samples to try
========================================================================

.. module:: producer
   :platform: Unix
   :synopsis: Suggest new parameter sets which optimize the objective.

"""
from abc import (ABC, abstractmethod)
import logging

from orion.core.io.database import DuplicateKeyError
from orion.core.utils import format_trials
from orion.core.worker.trials_history import TrialsHistory

log = logging.getLogger(__name__)


def get_objective(trial):
    objectives = [result['value'] for result in trial.results
                  if result['type'] == 'objective']

    if len(objectives) > 1:
        raise RuntimeError("Trial %d has %d objectives", trial.id, len(objectives))

    return objectives[0]


class BaseParallelStrategy(ABC):
    @abstractmethod
    def observe(self, trials):
        """observe completed trials"""
        pass

    @abstractmethod
    def lie(self, trials):
        """construct fake results for uncompleted trials"""
        pass

class NoParallelStrategy(BaseParallelStrategy):
    def observe(self, trials):
        pass

    def lie(self, trials):
        pass


class MaxParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        super(MaxParallelStrategy, self).observe(points, results)
        self.max_result = max(result['objective'] for result in results)

    def lie(self, trials):
        for trial in trials:
            if get_objective(trial):
                raise RuntimeError("Trial %d is completed but should not be.", trial.id)

            trial.results.append(Trial.Result(name='lie', type='lie', value=self.max_result))

        return trials


class MeanParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        super(MeanParallelStrategy, self).observe(points, results)
        self.mean_result = sum(result['objective'] for result in results) / float(len(results))

    def lie(self, trials):
        for trial in trials:
            if get_objective(trial):
                raise RuntimeError("Trial %d is completed but should not be.", trial.id)

            trial.results.append(Trial.Result(name='lie', type='lie', value=self.mean_result))

        return trials


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
        self.parallel_strategy = experiment.parallel_strategy  # TODO: Where should we define this
        self.naive_algorithm = None
        # TODO: Move trials_history into PrimaryAlgo during the refactoring of Algorithm with
        #       Strategist and Scheduler.
        self.trials_history = TrialsHistory()
        self.naive_trials_history = None

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

            new_points = self.naive_algorithm.suggest(self.pool_size)

            for new_point in new_points:
                log.debug("#### Convert point to `Trial` object.")
                new_trial = format_trials.tuple_to_trial(new_point, self.space)
                try:
                    new_trial.parents = self.naive_trials_history.children
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
            self.trials_history.update(completed_trials)
            self.algorithm.observe(points, results)
            self.parallel_strategy.observe(points, results)

    def _produces_lies(self):
        log.debug("### Fetch trials to observe:")
        non_completed_trials = self.experiment.fetch_active_trials()
        log.debug("### %s", non_completed_trials)
        if non_completed_trials:
            log.debug("### Use defined ParallelStrategy to assign them fake results.")
            lying_trials = self.parallel_strategy.lie(non_completed_trials)
            for lying_trial in lying_trials:
                log.debug("### Register to database: %s", lying_trial)
                lying_trial.parents = self.trials_history.children
                self.experiment.register_trial(lying_trial)

        return lying_trials

    def _update_naive_algorithm(self):
        """Pull all non completed trials to update naive model."""
        self.naive_algorithm = copy.deepcopy(self.algorithm)
        self.naive_trials_history = copy.deepcopy(self.trials_history)
        log.debug("### Create fake trials to observe:")
        lying_trials = self._produces_lies()
        log.debug("### %s", lying_trials)
        if lying_trials:
            log.debug("### Convert them to list of points and their results.")
            points = list(map(lambda trial: format_trials.trial_to_tuple(trial, self.space),
                              lying_trials))
            results = list(map(format_trials.get_trial_results, lying_trials))

            log.debug("### Observe them.")
            self.naive_trials_history.update(lying_trials)
            self.naive_algorithm.observe(points, results)
