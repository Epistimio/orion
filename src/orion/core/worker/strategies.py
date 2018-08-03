# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.strategies` -- register objectives for uncompleted trials
========================================================================

.. module:: strategies
   :platform: Unix
   :synopsis: Strategies to register objectives for uncompleted trials.

"""
from abc import (ABC, abstractmethod)
import logging

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
