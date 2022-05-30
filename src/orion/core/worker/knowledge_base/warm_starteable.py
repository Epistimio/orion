from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatch
import inspect

from orion.algo.base import BaseAlgorithm, algo_factory
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoWrapper
from orion.core.worker.knowledge_base.base import ExperimentInfo
from orion.core.worker.trial import Trial
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class WarmStarteable(Protocol):
    """Base class for Algorithms which can leverage 'related' past trials to bootstrap
    their optimization process.

    This is a protocol: It can be inherited from explicitly just like an ABC, but it can also be
    inherited from by just having a `warm_start` attribute.
    """

    @abstractmethod
    def warm_start(self, warm_start_trials: dict[ExperimentInfo, list[Trial]]):
        """Use the given trials to warm-start the algorithm.

        These experiments and their trials were fetched from some knowledge base, and
        are believed to be somewhat similar to the current on-going experiment.

        It is the responsibility of the Algorithm to implement this method in order to
        take advantage of these points.

        Parameters
        ----------
        warm_start_trials : Dict[Mapping, List[Trial]]
            Dictionary mapping from ExperimentInfo objects (dataclasses containing the
            experiment config) to the list of Trials associated with that experiment.
        """
        raise NotImplementedError(
            f"Algorithm of type {type(self)} isn't warm-starteable yet."
        )


@singledispatch
def is_warmstarteable(algo: BaseAlgorithm | type[BaseAlgorithm]) -> bool:
    """Returns whether the given algo, algo type, or algo config, supports warm-starting.

    Parameters
    ----------
    algo : Union[BaseAlgorithm, Type[BaseAlgorithm], Dict[str, Any]]
        Algorithm instance, type of algorithm, or algorithm configuration dictionary.

    Returns
    -------
    bool
        Whether the input is or describes a warm-starteable algorithm.
    """
    raise NotImplementedError(
        f"Don't know how to tell if {algo} is a warm-starteable algorithm."
    )


@is_warmstarteable.register(BaseAlgorithm)
def _(algo: BaseAlgorithm) -> bool:
    return isinstance(algo, WarmStarteable)


@is_warmstarteable.register(type)
def _(algo: type[BaseAlgorithm]) -> bool:
    return issubclass(algo, WarmStarteable)


@is_warmstarteable.register(AlgoWrapper)
def _(algo: AlgoWrapper) -> bool:
    # NOTE: Not directing directly to algo.unwrapped, because the MultiTask wrapper makes the algo
    # WarmStarteable. Move down the chain of wrappers instead.
    return is_warmstarteable(algo) or is_warmstarteable(algo.algorithm)


@is_warmstarteable.register(str)
def _(algo: str) -> bool:
    # NOTE: I don't think the algorithm should ever be a string, since now we pass the algo class
    # in create_algo. Adding this here in case it comes in handy at some point.
    algo_type = algo_factory.get_class(algo)
    return is_warmstarteable(algo_type)
