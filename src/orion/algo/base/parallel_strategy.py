"""
Parallel Strategies
===================

Register objectives for incomplete trials.

Parallel strategy objects can be created using `strategy_factory.create('strategy_name')`.

"""
from __future__ import annotations

import copy
import logging

from orion.algo.base.registry import Registry
from orion.core.utils import GenericFactory
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


CORRUPTED_DB_WARNING = """\
Trial `%s` has an objective but status is not completed.
This is likely due to a corrupted database, possibly because of
database timeouts. Try setting manually status to `completed`.
You can find documentation to do this at
https://orion.readthedocs.io/en/stable/user/storage.html#storage-backend.

If you encounter this issue often, please consider reporting it to
https://github.com/Epistimio/orion/issues."""


def get_objective(trial: Trial) -> float | None:
    """Get the value for the objective, if it exists, for this trial

    :return: Float or None
        The value of the objective, or None if it doesn't exist
    """
    objectives = [
        result.value for result in trial.results if result.type == "objective"
    ]
    objective: float | None = None
    if not objectives:
        objective = None
    elif len(objectives) == 1:
        objective = objectives[0]
    elif len(objectives) > 1:
        raise RuntimeError(f"Trial {trial.id} has {len(objectives)} objectives")

    return objective


# TODO: Should add a strategy for broken trials.
# TODO: has_observed from algorithms should return True for broken trials.
# TODO: Default


# We want stub parallel strategy for Hyperband/ASHA/TPE for broken
# We want MaxParallelStrategy for TPE.
# It is so algorithm dependent, it should be within the algorithms.
# strategy:
#   broken:
#       StubParallelStrategy:
#           stub_value: 10000
#   else:
#       MeanParallelStrategy:
#           default_result: 0.5


class ParallelStrategy:
    """Strategy to give intermediate results for incomplete trials"""

    def __init__(self):
        self.registry = Registry()

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the strategy."""
        return {"registry": self.registry.state_dict}

    # pylint: disable=missing-function-docstring
    def set_state(self, state_dict: dict) -> None:
        self.registry.set_state(state_dict["registry"])

    def observe(self, trials: list[Trial]) -> None:
        """Observe completed trials

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe` method

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            self.registry.register(trial)

    # pylint: disable=missing-function-docstring
    def infer(self, trial: Trial) -> Trial | None:
        fake_result = self.lie(trial)
        if fake_result is None:
            return None

        fake_trial = copy.deepcopy(trial)

        # pylint: disable=protected-access
        fake_trial._results.append(fake_result)
        return fake_trial

    def lie(self, trial: Trial) -> Trial.Result | None:
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
    def configuration(self) -> dict:
        """Provide the configuration of the strategy as a dictionary."""
        return {"of_type": self.__class__.__name__.lower()}


class NoParallelStrategy(ParallelStrategy):
    """No parallel strategy"""

    def lie(self, trial: Trial) -> Trial.Result | None:
        """See ParallelStrategy.lie"""
        result = super().lie(trial)
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

    def __init__(
        self, strategy_configs: dict | None = None, default_strategy: dict | None = None
    ):
        super().__init__()
        if strategy_configs is None:
            strategy_configs = {"broken": {"of_type": "maxparallelstrategy"}}

        self.strategies = {}
        for status, strategy_config in strategy_configs.items():
            self.strategies[status] = strategy_factory.create(**strategy_config)

        if default_strategy is None:
            default_strategy = {"of_type": "noparallelstrategy"}

        self.default_strategy: ParallelStrategy = strategy_factory.create(
            **default_strategy
        )

    @property
    def configuration(self) -> dict:
        configuration = super().configuration
        configuration["strategy_configs"] = {
            status: strategy.configuration
            for status, strategy in self.strategies.items()
        }
        configuration["default_strategy"] = self.default_strategy.configuration

        return configuration

    @property
    def state_dict(self) -> dict:
        state_dict = super().state_dict
        state_dict["strategies"] = {
            status: strategy.state_dict for status, strategy in self.strategies.items()
        }
        state_dict["default_strategy"] = self.default_strategy.state_dict
        return state_dict

    # pylint: disable=missing-function-docstring
    def set_state(self, state_dict: dict) -> None:
        super().set_state(state_dict)

        # pylint: disable=consider-using-dict-items
        for status in self.strategies:
            self.strategies[status].set_state(state_dict["strategies"][status])
        self.default_strategy.set_state(state_dict["default_strategy"])

    # pylint: disable=missing-function-docstring
    def get_strategy(self, trial: Trial) -> ParallelStrategy:
        strategy = self.strategies.get(trial.status)

        if strategy is None:
            return self.default_strategy

        return strategy

    def observe(self, trials: list[Trial]) -> None:
        for trial in trials:
            for strategy in self.strategies.values():
                strategy.observe([trial])
            self.default_strategy.observe([trial])

    def lie(self, trial: Trial) -> Trial.Result | None:
        # print(
        #     trial.status, self.get_strategy(trial), self.get_strategy(trial).max_result
        # )
        return self.get_strategy(trial).lie(trial)


class MaxParallelStrategy(ParallelStrategy):
    """Parallel strategy that uses the max of completed objectives"""

    def __init__(self, default_result=float("inf")):
        """Initialize the maximum result used to lie"""
        super().__init__()
        self.default_result = default_result

    @property
    def configuration(self) -> dict:
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super().configuration
        configuration["default_result"] = self.default_result
        return configuration

    # pylint: disable=missing-function-docstring
    @property
    def max_result(self) -> float:
        objectives = [
            trial.objective.value
            for trial in self.registry
            if trial.status == "completed" and trial.objective is not None
        ]
        if not objectives:
            return self.default_result
        return max(objectives)

    def lie(self, trial: Trial) -> Trial.Result | None:
        """See ParallelStrategy.lie"""
        result = super().lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.max_result)


class MeanParallelStrategy(ParallelStrategy):
    """Parallel strategy that uses the mean of completed objectives"""

    def __init__(self, default_result: float = float("inf")):
        """Initialize the mean result used to lie"""
        super().__init__()
        self.default_result = default_result

    @property
    def configuration(self) -> dict:
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super().configuration
        configuration["default_result"] = self.default_result
        return configuration

    @property
    def mean_result(self) -> float:
        """Compute the mean result"""
        objectives = [
            trial.objective.value
            for trial in self.registry
            if trial.status == "completed" and trial.objective is not None
        ]
        if not objectives:
            return self.default_result
        return sum(objectives) / len(objectives)

    def lie(self, trial: Trial) -> Trial.Result | None:
        """See ParallelStrategy.lie"""
        result = super().lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.mean_result)


class StubParallelStrategy(ParallelStrategy):
    """Parallel strategy that returns static objective value for incompleted trials."""

    def __init__(self, stub_value: float | None = None):
        """Initialize the stub value"""
        super().__init__()
        self.stub_value = stub_value

    @property
    def configuration(self) -> dict:
        """Provide the configuration of the strategy as a dictionary."""
        configuration = super().configuration
        configuration["stub_value"] = self.stub_value
        return configuration

    def lie(self, trial: Trial) -> Trial.Result | None:
        """See ParallelStrategy.lie"""
        result = super().lie(trial)
        if result:
            return result

        return Trial.Result(name="lie", type="objective", value=self.stub_value)


strategy_factory = GenericFactory(ParallelStrategy)
