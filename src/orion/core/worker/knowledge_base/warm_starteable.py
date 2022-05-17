from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any

from orion.algo.base import BaseAlgorithm, algo_factory
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoWrapper
from orion.core.worker.knowledge_base.base import ExperimentInfo
from orion.core.worker.trial import Trial


class WarmStarteable(ABC):
    """Base class for Algorithms which can leverage 'related' past trials to bootstrap
    their optimization process.
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

    # @contextmanager
    # @abstractmethod
    # def warm_start_mode(self):
    #     pass


@singledispatch
def is_warmstarteable(
    algo: BaseAlgorithm | type[BaseAlgorithm] | dict[str, Any]
) -> bool:
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
    return False


@is_warmstarteable.register(type)
def _(algo: type[BaseAlgorithm]) -> bool:
    return issubclass(algo, WarmStarteable)


@is_warmstarteable.register(BaseAlgorithm)
def _(algo: BaseAlgorithm) -> bool:
    return isinstance(algo, WarmStarteable)


@is_warmstarteable.register(AlgoWrapper)
def _(algo: AlgoWrapper) -> bool:
    # NOTE: Not directing directly to algo.unwrapped, because there might eventually be
    # a Wrapper that makes an algo WarmStarteable. Recurse down the chain of wrappers
    # instead.
    return is_warmstarteable(algo.algorithm)


@is_warmstarteable.register(str)
def _(algo: str) -> bool:
    available_algos = algo_factory.get_classes()
    if algo not in available_algos and algo.lower() not in available_algos:
        raise RuntimeError(
            f"Can't tell if algo '{algo}'' is warm-starteable, since there are no "
            f"algos registered with that name!\n"
            f"Available algos: {list(available_algos.keys())}"
        )
    algo_type = available_algos.get(algo, available_algos.get(algo.lower()))
    assert algo_type is not None
    return inspect.isclass(algo_type) and issubclass(algo_type, WarmStarteable)


@is_warmstarteable.register(dict)
def _(algo: dict[str, Any]) -> bool:
    first_key = list(algo)[0]
    return len(algo) == 1 and is_warmstarteable(first_key)
