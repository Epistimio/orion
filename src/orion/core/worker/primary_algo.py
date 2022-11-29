""" Utility function for creating and wrapping the HPO algorithm. """
from __future__ import annotations

import inspect
from typing import Callable, TypeVar, cast, overload

from typing_extensions import Concatenate, ParamSpec

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers import InsistSuggest
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform
from orion.core.worker.warm_start import KnowledgeBase, MultiTaskWrapper
from orion.core.worker.warm_start.warm_starteable import (
    WarmStarteable,
    is_warmstarteable,
)

AlgoT = TypeVar("AlgoT", bound=BaseAlgorithm)
WarmStarteableAlgoT = TypeVar("WarmStarteableAlgoT", bound=WarmStarteable)

P = ParamSpec("P")


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], AlgoT],
    space: Space,
    knowledge_base: None = None,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> InsistSuggest[SpaceTransform[AlgoT]]:
    """Creates an algorithm of the given type."""


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], WarmStarteableAlgoT],
    space: Space,
    knowledge_base: KnowledgeBase,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> InsistSuggest[SpaceTransform[WarmStarteableAlgoT]]:
    ...


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], AlgoT],
    space: Space,
    knowledge_base: KnowledgeBase,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> MultiTaskWrapper[InsistSuggest[SpaceTransform[AlgoT]]]:
    ...


def create_algo(
    algo_type: Callable[Concatenate[Space, P], AlgoT],
    space: Space,
    knowledge_base: KnowledgeBase | None = None,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> InsistSuggest[SpaceTransform[AlgoT]] | MultiTaskWrapper[
    InsistSuggest[SpaceTransform[AlgoT]]
]:
    """Adds different wrappers on top of an algorithm of type `algo_type` before it gets used.

    These wrappers are used to:
    - apply the transformations required for the algorithm to be applied to the given search space;
    - If a knowledge base is passed, and the algorithm isn't warmstarteable, the algo is wrapped
      with a `MultiTaskWrapper`;
    - Make sure that calls to returned algo's `suggest` method returns a Trial, by trying a few
      times.

    Parameters
    ----------
    algo_type
        Type of algorithm to create and wrap.
    space
        The (original, un-transformed) search space.
    knowledge_base
        The Knowledge base to use for warm-starting of the algorithm, by default None.

    Returns
    -------
        An AlgoWrapper around the algorithm of type `algo_type`.
    """
    if not (inspect.isclass(algo_type) and issubclass(algo_type, BaseAlgorithm)):
        raise RuntimeError(
            f"algo_type must be a type of algorithm (a subclass of BaseAlgorithm), not {algo_type}"
        )
    spaces = [space]
    # Create the spaces for each wrapper, from the top down.
    if knowledge_base and not is_warmstarteable(algo_type):
        space = MultiTaskWrapper.transform_space(space, knowledge_base=knowledge_base)
        spaces.append(space)

    space = InsistSuggest.transform_space(space)
    spaces.append(space)

    space = SpaceTransform.transform_space(space, algo_type=algo_type)
    spaces.append(space)
    # Create the algo, using the innermost (most transformed) space.
    # Then, create each wrapper, from the bottom up.
    algorithm = algo_type(spaces.pop(), *algo_args, **algo_kwargs)
    # NOTE: type cast is needed temporarily, because the (Pylance) type checker incorrectly assumes
    # that the type of `algo_type` has been narrowed to `type[BaseAlgorithm]` by the if/raise
    # above, but it is in fact a subclass of BaseAlgorithm (type[AlgoT]).
    algorithm = cast(AlgoT, algorithm)
    algorithm = SpaceTransform(space=spaces.pop(), algorithm=algorithm)
    algorithm = InsistSuggest(space=spaces.pop(), algorithm=algorithm)
    if knowledge_base and not is_warmstarteable(algo_type):
        algorithm = MultiTaskWrapper(space=spaces.pop(), algorithm=algorithm)
    return algorithm
