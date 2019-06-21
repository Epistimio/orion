# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.strategy` -- register objectives for incomplete trials
========================================================================

.. module:: strategy
   :platform: Unix
   :synopsis: Strategies to register objectives for incomplete trials.

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import Factory
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


def get_objective(trial):
    """Get the value for the objective, if it exists, for this trial

    :return: Float or None
        The value of the objective, or None if it doesn't exist
    """
    objectives = [result['value'] for result in trial.results
                  if result['type'] == 'objective']

    if not objectives:
        objective = None
    elif len(objectives) == 1:
        objective = objectives[0]
    elif len(objectives) > 1:
        raise RuntimeError("Trial {} has {} objectives".format(trial.id, len(objectives)))

    return objective


class BaseParallelStrategy(object, metaclass=ABCMeta):
    """Strategy to give intermediate results for incomplete trials"""

    @abstractmethod
    def observe(self, points, results):
        """Observe completed trials

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe` method

        :param points: list of tuples of array-likes
           Points from a `orion.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        :param results : list of dicts
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.
        """
        # NOTE: In future points and results will be converted to trials for coherence with
        # `Strategy.lie()` as well as for coherence with `Algorithm.observe` which will also be
        # converted to expect trials instead of lists and dictionaries.
        pass

    @abstractmethod
    def lie(self, trial):
        """Construct a fake result for an incomplete trial

        :param trial: `orion.core.worker.trial.Trial`
        :return: Float or None
            The fake objective result corresponding to the trial given
        """
        pass

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        # TODO(mnoukhov): change to dict {of_type: __name__} ?
        return self.__class__.__name__


class NoParallelStrategy(BaseParallelStrategy):
    """No parallel strategy"""

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        pass


class MaxParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that uses the max of completed objectives"""

    def __init__(self, default_result=float('inf')):
        """Initialize the maximum result used to lie"""
        self.max_result = default_result

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MaxParallelStrategy, self).observe(points, results)
        self.max_result = max(result['objective'] for result in results
                              if result['objective'] is not None)

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if get_objective(trial):
            raise RuntimeError("Trial {} is completed but should not be.".format(trial.id))

        return Trial.Result(name='lie', type='lie', value=self.max_result)


class MeanParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that uses the mean of completed objectives"""

    def __init__(self, default_result=float('inf')):
        """Initialize the mean result used to lie"""
        self.mean_result = default_result

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MeanParallelStrategy, self).observe(points, results)
        objective_values = [result['objective'] for result in results
                            if result['objective'] is not None]
        self.mean_result = sum(value for value in objective_values) / float(len(objective_values))

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if get_objective(trial):
            raise RuntimeError("Trial {} is completed but should not be.".format(trial.id))

        return Trial.Result(name='lie', type='lie', value=self.mean_result)


class StubParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that returns static objective value for incompleted trials."""

    def __init__(self, stub_value=None):
        """Initialize the stub value"""
        self.stub_value = stub_value

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if get_objective(trial):
            raise RuntimeError("Trial {} is completed but should not be.".format(trial.id))

        return Trial.Result(name='lie', type='lie', value=self.stub_value)


# pylint: disable=too-few-public-methods,abstract-method
class Strategy(BaseParallelStrategy, metaclass=Factory):
    """Class used to build a parallel strategy given name and params

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseParallelStrategy` interface.
    """

    pass
