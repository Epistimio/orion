# -*- coding: utf-8 -*-
"""
Parallel Strategies
===================

Register objectives for incomplete trials.

Parallel strategy objects can be created using `strategy_factory.create('strategy_name')`.

"""
import copy
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


# TODO: Should add a strategy for broken trials.
# TODO: has_observed from algorithms should return True for broken trials.
# TODO: Default


# We want stub parallel strategy for Hyperband/ASHA/TPE for broken
# We want MaxParallelStrategy for TPE.
# It is so algorithm dependant, it should be within the algorithms.
# strategy:
#   broken:
#       StubParallelStrategy:
#           stub_value: 10000
#   else:
#       MeanParallelStrategy:
#           default_result: 0.5


class ParallelStrategy(object):
    """Strategy to give intermediate results for incomplete trials"""

    def __init__(self, *args, **kwargs):
        self._trials_info = {}

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the strategy."""
        return {"_trials_info": self._trials_info}

    def set_state(self, state_dict):
        self._trials_info = state_dict["_trials_info"]

    def observe(self, trials):
        """Observe completed trials

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe` method

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            self._trials_info[trial.id] = trial

    def infer(self, trial):
        fake_result = self.lie(trial)
        if fake_result is None:
            return None

        fake_trial = copy.deepcopy(trial)
        fake_trial._results.append(fake_result)
        return fake_trial

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
            return Trial.Result(name="lie", type="objective", value=objective)

        return None

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        return {"of_type": self.__class__.__name__.lower()}


class NoParallelStrategy(ParallelStrategy):
    """No parallel strategy"""

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(NoParallelStrategy, self).lie(trial)
        if result:
            return result

        return None


class StatusBasedParallelStrategy(ParallelStrategy):
    """Different parallel strategies for different trial status

    Parameters
    ----------
    strategy_configs: dict
        Dictionary of strategy configurations. Each key should be a valid
        trial status.
    default_strategy: dict or None, optional
        Default strategy for trial status that are not defined by ``strategy_configs``.
        Default is NoParallelStrategy(), which always returns None.
    """

    def __init__(self, strategy_configs=None, default_strategy=None):
        super(StatusBasedParallelStrategy, self).__init__()
        if strategy_configs is None:
            strategy_configs = {"broken": {"of_type": "maxparallelstrategy"}}

        self.strategies = dict()
        for status, strategy_config in strategy_configs.items():
            self.strategies[status] = strategy_factory.create(**strategy_config)

        if default_strategy is None:
            default_strategy = {"of_type": "noparallelstrategy"}

        self.default_strategy = strategy_factory.create(**default_strategy)

    @property
    def configuration(self):
        configuration = super(StatusBasedParallelStrategy, self).configuration
        configuration["strategy_configs"] = {
            status: strategy.configuration
            for status, strategy in self.strategies.items()
        }
        configuration["default_strategy"] = self.default_strategy.configuration

        return configuration

    @property
    def state_dict(self):
        state_dict = super(StatusBasedParallelStrategy, self).state_dict
        state_dict["strategies"] = {
            status: strategy.state_dict for status, strategy in self.strategies.items()
        }
        state_dict["default_strategy"] = self.default_strategy.state_dict
        return state_dict

    def set_state(self, state_dict):
        super(StatusBasedParallelStrategy, self).set_state(state_dict)
        for status in self.strategies.keys():
            self.strategies[status].set_state(state_dict["strategies"][status])
        self.default_strategy.set_state(state_dict["default_strategy"])

    def get_strategy(self, trial):
        strategy = self.strategies.get(trial.status)

        if strategy is None:
            return self.default_strategy

        return strategy

    def observe(self, trials):
        for trial in trials:
            for strategy in self.strategies.values():
                strategy.observe([trial])
            self.default_strategy.observe([trial])

    def lie(self, trial):
        # print(
        #     trial.status, self.get_strategy(trial), self.get_strategy(trial).max_result
        # )
        return self.get_strategy(trial).lie(trial)


class MaxParallelStrategy(ParallelStrategy):
    """Parallel strategy that uses the max of completed objectives"""

    def __init__(self, default_result=float("inf")):
        """Initialize the maximum result used to lie"""
        super(MaxParallelStrategy, self).__init__()
        self.default_result = default_result

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super(MaxParallelStrategy, self).configuration
        configuration["default_result"] = self.default_result
        return configuration

    @property
    def max_result(self):
        objectives = [
            trial.objective.value
            for trial in self._trials_info.values()
            if trial.status == "completed"
        ]
        if not objectives:
            return self.default_result
        return max(objectives)

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(MaxParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.max_result)


class MeanParallelStrategy(ParallelStrategy):
    """Parallel strategy that uses the mean of completed objectives"""

    def __init__(self, default_result=float("inf")):
        """Initialize the mean result used to lie"""
        super(MeanParallelStrategy, self).__init__()
        self.default_result = default_result

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super(MeanParallelStrategy, self).configuration
        configuration["default_result"] = self.default_result
        return configuration

    @property
    def mean_result(self):
        objectives = [
            trial.objective.value
            for trial in self._trials_info.values()
            if trial.status == "completed"
        ]
        if not objectives:
            return self.default_result
        return sum(objectives) / len(objectives)

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(MeanParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.mean_result)


class StubParallelStrategy(ParallelStrategy):
    """Parallel strategy that returns static objective value for incompleted trials."""

    def __init__(self, stub_value=None):
        """Initialize the stub value"""
        super(StubParallelStrategy, self).__init__()
        self.stub_value = stub_value

    @property
    def configuration(self):
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super(StubParallelStrategy, self).configuration
        configuration["stub_value"] = self.stub_value
        return configuration

    def lie(self, trial):
        """See ParallelStrategy.lie"""
        result = super(StubParallelStrategy, self).lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.stub_value)


strategy_factory = GenericFactory(ParallelStrategy)
