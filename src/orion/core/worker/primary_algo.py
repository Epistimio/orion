from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

from typing_extensions import Concatenate, ParamSpec

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers import InsistSuggest, SpaceTransform

# Backward compatibility imports adapters.
from orion.core.worker.algo_wrappers.space_transform import SpaceTransform  # noqa
from orion.core.worker.warm_start import KnowledgeBase, MultiTaskWrapper
from orion.core.worker.warm_start.warm_starteable import WarmStarteable

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)
WarmStarteableAlgo = TypeVar("WarmStarteableAlgo", bound=WarmStarteable)

P = ParamSpec("P")


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], AlgoType],
    space: Space,
    knowledge_base: KnowledgeBase,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> MultiTaskWrapper[InsistSuggest[SpaceTransform[AlgoType]]]:
    ...


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], WarmStarteableAlgo],
    space: Space,
    knowledge_base: KnowledgeBase,
    *algo_args,
    **algo_kwargs,
) -> InsistSuggest[SpaceTransform[WarmStarteableAlgo]]:
    ...


@overload
def create_algo(
    algo_type: Callable[Concatenate[Space, P], AlgoType],
    space: Space,
    knowledge_base: None = None,
    *algo_args: P.args,
    **algo_kwargs: P.kwargs,
) -> InsistSuggest[SpaceTransform[AlgoType]]:
    ...


def create_algo(
    algo_type: Callable[..., AlgoType],
    space: Space,
    knowledge_base: KnowledgeBase | None = None,
    *algo_args: Any,
    **algo_kwargs: Any,
) -> InsistSuggest[SpaceTransform[AlgoType]] | MultiTaskWrapper[
    InsistSuggest[SpaceTransform[AlgoType]]
]:
    """Adds different wrappers on top of an algorithm of type `algo_type` before it gets used.

    These wrappers are used to:
    - apply the transformations required for the algorithm to be applied to the given search space;
    - check whether the algorithm is warmstarteable and wrap it in a `MultiTaskWrapper` if it
      isn't;
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
    spaces = [space]
    # Create the spaces for each wrapper, from the top down.
    if knowledge_base:
        space = MultiTaskWrapper.transform_space(space, knowledge_base=knowledge_base)
        spaces.append(space)

    space = InsistSuggest.transform_space(space)
    spaces.append(space)

    space = SpaceTransform.transform_space(space, algo_type=algo_type)
    spaces.append(space)

    # Create the algo, using the innermost (most transformed) space.
    # Then, create each wrapper, from the bottom up.
    algorithm = algo_type(space=spaces.pop(), *algo_args, **algo_kwargs)
    algorithm = SpaceTransform(space=spaces.pop(), algorithm=algorithm)
    algorithm = InsistSuggest(space=spaces.pop(), algorithm=algorithm)
    if knowledge_base:
        algorithm = MultiTaskWrapper(space=spaces.pop(), algorithm=algorithm)
    return algorithm
