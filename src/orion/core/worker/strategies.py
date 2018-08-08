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
    @abstractmethod
    def observe(self, points, results):
        """observe completed trials"""
        pass

    @abstractmethod
    def lie(self, trials):
        """construct a fake result for an incomplete trial"""
        pass


class NoParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        pass

    def lie(self, trial):
        pass


class MaxParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        super(MaxParallelStrategy, self).observe(points, results)
        #TODO(mnoukhov): observe all types or just objective?
        self.max_result = max(result.value for result in results
                              if result.type == 'objective')

    def lie(self, trial):
        if get_objective(trial):
            raise RuntimeError("Trial %d is completed but should not be.", trial.id)

        return Trial.Result(name='lie', type='lie', value=self.max_result)


class MeanParallelStrategy(BaseParallelStrategy):
    def observe(self, points, results):
        super(MeanParallelStrategy, self).observe(points, results)
        objective_values = [result.value for result in results if result.type == 'objective']
        self.mean_result = sum(value for value in objective_values) / float(len(objective_values))

    def lie(self, trial):
        if get_objective(trial):
            raise RuntimeError("Trial %d is completed but should not be.", trial.id)

        return Trial.Result(name='lie', type='lie', value=self.mean_result)


class Strategy(BaseParallelStrategy, metaclass=Factory):
    """Class used to inject dependency on an adapter implementation.

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseAlgorithm` interface.
    """

    @classmethod
    def build(cls, strategy_dict):
        """Builder method for an adapter

        Parameters
        ----------
        strategy_dict: dict
            Strategy representation in dict form

        Returns
        -------
        `orion.core.worker.BaseParallelStrategy`
            A parallel strategy class

        """
        return cls(**strategy_dict)
