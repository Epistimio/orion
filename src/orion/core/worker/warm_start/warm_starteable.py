""" ABC for a warm-starteable algorithm. """
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import overload

from typing_extensions import TypeGuard

from orion.algo.base import BaseAlgorithm, algo_factory
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoWrapper
from orion.core.worker.experiment_config import ExperimentConfig
from orion.core.worker.trial import Trial


class WarmStarteable(BaseAlgorithm, ABC):
    """Base class for Algorithms which can leverage 'related' past trials to bootstrap
    their optimization process.
    """

    @abstractmethod
    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentConfig, list[Trial]]]
    ) -> None:
        """Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsibility of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials : Dict[Mapping, List[Trial]]
            Dictionary mapping from ExperimentConfig objects (dataclasses containing the
            experiment config) to the list of Trials associated with that experiment.
        """
        raise NotImplementedError(
            f"Algorithm of type {type(self)} isn't warm-starteable yet."
        )


@overload
def is_warmstarteable(algo: type[BaseAlgorithm]) -> TypeGuard[type[WarmStarteable]]:
    ...


@overload
def is_warmstarteable(algo: BaseAlgorithm) -> TypeGuard[WarmStarteable]:
    ...


@overload
def is_warmstarteable(
    algo: AlgoWrapper,
) -> TypeGuard[WarmStarteable | AlgoWrapper[WarmStarteable]]:
    ...


@overload
def is_warmstarteable(algo: str) -> bool:
    ...


def is_warmstarteable(
    algo: BaseAlgorithm | type[BaseAlgorithm] | str | AlgoWrapper,
) -> TypeGuard[
    WarmStarteable | AlgoWrapper[WarmStarteable] | type[WarmStarteable]
] | bool:
    """Returns whether the given algo type or instance supports warm-starting.

    Parameters
    ----------
    algo : Union[BaseAlgorithm, Type[BaseAlgorithm]]
        Algorithm type or instance.

    Returns
    -------
    bool
        Whether the input is a warm-starteable algorithm.
    """
    if isinstance(algo, AlgoWrapper):
        # NOTE: Not going directly to algo.unwrapped, because the MultiTask wrapper makes the algo
        # WarmStarteable. Check down the chain of wrappers instead.
        return isinstance(algo, WarmStarteable) or is_warmstarteable(algo.algorithm)
    if isinstance(algo, BaseAlgorithm):
        return isinstance(algo, WarmStarteable)
    if isinstance(algo, str):
        # NOTE: I don't think the algorithm should ever be a string, since now we pass the algo
        # class in create_algo. Adding this here in case it comes in handy at some point.
        algo_type = algo_factory.get_class(algo)
        return is_warmstarteable(algo_type)
    if inspect.isclass(algo):
        return issubclass(algo, WarmStarteable)
    raise NotImplementedError(
        f"Don't know how to tell if {algo} is a warm-starteable algorithm."
    )
