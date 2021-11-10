# -*- coding: utf-8 -*-
"""
Parallel Strategies
===================

Register objectives for incomplete trials.

Parallel strategy objects can be created using `strategy_factory.create('strategy_name')`.

"""
import logging

from orion.core.utils import GenericFactory
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


class ParallelStrategy(object):
    """Strategy to give intermediate results for incomplete trials"""

    def __init__(self, *args, **kwargs):
        pass

    def observe(self, trials):
        """Observe completed trials

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe` method

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        raise NotImplementedError()

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


class NoParallelStrategy(ParallelStrategy):
    """No parallel strategy"""

    def observe(self, trials):
        """See ParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(NoParallelStrategy, self).lie(trial)
        if result:
            return result

        return None


class MaxParallelStrategy(ParallelStrategy):
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

    def observe(self, trials):
        """See ParallelStrategy.observe"""
        results = [
            trial.objective.value for trial in trials if trial.objective is not None
        ]
        if results:
            self.max_result = max(results)

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(MaxParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.max_result)


class MeanParallelStrategy(ParallelStrategy):
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

    def observe(self, trials):
        """See ParallelStrategy.observe"""
        objective_values = [
            trial.objective.value for trial in trials if trial.objective is not None
        ]
        if objective_values:
            self.mean_result = sum(value for value in objective_values) / float(
                len(objective_values)
            )

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(MeanParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.mean_result)


class StubParallelStrategy(ParallelStrategy):
    """Parallel strategy that returns static objective value for incompleted trials."""

    def __init__(self, stub_value=None):
        """Initialize the stub value"""
        super(StubParallelStrategy, self).__init__()
        self.stub_value = stub_value

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return {self.__class__.__name__: {"stub_value": self.stub_value}}

    def observe(self, trials):
        """See ParallelStrategy.observe"""
        pass

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(StubParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="lie", value=self.stub_value)


strategy_factory = GenericFactory(ParallelStrategy)
