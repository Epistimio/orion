# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.strategies` -- register objectives for incomplete trials
========================================================================

.. module:: strategies
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
        raise RuntimeError("Trial %d has %d objectives", trial.id, len(objectives))

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
        pass

    @abstractmethod
    def lie(self, trial):
        """Construct a fake result for an incomplete trial

        :param trial: `orion.core.worker.trial.Trial`
        :return: Float or None
            The fake objective result corresponding to the trial given
        """
        pass


class NoParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        pass


class MaxParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MaxParallelStrategy, self).observe(points, results)
        #TODO(mnoukhov): observe all types or just objective?
        self.max_result = max(result.value for result in results
                              if result.type == 'objective')

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if get_objective(trial):
            raise RuntimeError("Trial %d is completed but should not be.", trial.id)

        return Trial.Result(name='lie', type='lie', value=self.max_result)


class MeanParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MeanParallelStrategy, self).observe(points, results)
        objective_values = [result.value for result in results if result.type == 'objective']
        self.mean_result = sum(value for value in objective_values) / float(len(objective_values))

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if get_objective(trial):
            raise RuntimeError("Trial %d is completed but should not be.", trial.id)

        return Trial.Result(name='lie', type='lie', value=self.mean_result)


class Strategy(BaseParallelStrategy, metaclass=Factory):
    """Class used to build a parallel strategy given name and params

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseParallelStrategy` interface.
    """

    @classmethod
    def build(cls, strategy_dict):
        """Builder method for a parallel strategy

        :param strategy_dict: dict
            Strategy representation in dict form
        :return: `orion.core.worker.BaseParallelStrategy`
            An instantiated parallel strategy class
        """
        return cls(**strategy_dict)
