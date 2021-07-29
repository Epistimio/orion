# -*- coding: utf-8 -*-
"""
Parallel Strategies
===================

Register objectives for incomplete trials

"""
import logging
from abc import ABCMeta, abstractmethod

from orion.core.utils import Factory
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


CORRUPTED_DB_WARNING = """\
Trial `%s` has an objective but status is not completed.
This is likely due to a corrupted database, possibly because of
database timeouts. Try setting manually status to `completed`.
You can find documention to do this at
https://orion.readthedocs.io/en/stable/user/storage.html#storage-backend.

If you encounter this issue often, please consider reporting it to
https://github.com/Epistimio/orion/issues."""


def get_objective(trial):
    """Get the value for the objective, if it exists, for this trial

    :return: Float or None
        The value of the objective, or None if it doesn't exist
    """
    objectives = [
        result.value for result in trial.results if result.type == "objective"
    ]

    if not objectives:
        objective = None
    elif len(objectives) == 1:
        objective = objectives[0]
    elif len(objectives) > 1:
        raise RuntimeError(
            "Trial {} has {} objectives".format(trial.id, len(objectives))
        )

    return objective


class BaseParallelStrategy(object, metaclass=ABCMeta):
    """Strategy to give intermediate results for incomplete trials"""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def observe(self, points, results):
        """Observe completed trials

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe` method

        Parameters
        ----------
        points: list of tuples of array-likes
           Points from a `orion.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        results: list of dict
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.

        """
        # NOTE: In future points and results will be converted to trials for coherence with
        # `Strategy.lie()` as well as for coherence with `Algorithm.observe` which will also be
        # converted to expect trials instead of lists and dictionaries.
        pass

    # pylint: disable=no-self-use
    def lie(self, trial):
        """Construct a fake result for an incomplete trial

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            A trial object which is not supposed to be completed.

        Returns
        -------
        ``orion.core.worker.trial.Trial.Result``
            The fake objective result corresponding to the trial given.

        Notes
        -----
        If the trial has an objective even if not completed, a warning is printed to user
        with a pointer to documentation to resolve the database corruption. The result returned is
        the corresponding objective instead of the lie.

        """
        objective = get_objective(trial)
        if objective:
            log.warning(CORRUPTED_DB_WARNING, trial.id)
            return Trial.Result(name="lie", type="lie", value=objective)

        return None

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return self.__class__.__name__


class NoParallelStrategy(BaseParallelStrategy):
    """No parallel strategy"""

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        result = super(NoParallelStrategy, self).lie(trial)
        if result:
            return result

        return None


class MaxParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that uses the max of completed objectives"""

    def __init__(self, default_result=float("inf")):
        """Initialize the maximum result used to lie"""
        super(MaxParallelStrategy, self).__init__()
        self.default_result = default_result
        self.max_result = default_result

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return {self.__class__.__name__: {"default_result": self.default_result}}

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MaxParallelStrategy, self).observe(points, results)
        results = [
            result["objective"] for result in results if result["objective"] is not None
        ]
        if results:
            self.max_result = max(results)

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        result = super(MaxParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.max_result)


class MeanParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that uses the mean of completed objectives"""

    def __init__(self, default_result=float("inf")):
        """Initialize the mean result used to lie"""
        super(MeanParallelStrategy, self).__init__()
        self.default_result = default_result
        self.mean_result = default_result

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return {self.__class__.__name__: {"default_result": self.default_result}}

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        super(MeanParallelStrategy, self).observe(points, results)
        objective_values = [
            result["objective"] for result in results if result["objective"] is not None
        ]
        if objective_values:
            self.mean_result = sum(value for value in objective_values) / float(
                len(objective_values)
            )

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        result = super(MeanParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.mean_result)


class StubParallelStrategy(BaseParallelStrategy):
    """Parallel strategy that returns static objective value for incompleted trials."""

    def __init__(self, stub_value=None):
        """Initialize the stub value"""
        super(StubParallelStrategy, self).__init__()
        self.stub_value = stub_value

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return {self.__class__.__name__: {"stub_value": self.stub_value}}

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        result = super(StubParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.stub_value)


# pylint: disable=too-few-public-methods,abstract-method
class Strategy(BaseParallelStrategy, metaclass=Factory):
    """Class used to build a parallel strategy given name and params

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseParallelStrategy` interface.
    """

    pass
