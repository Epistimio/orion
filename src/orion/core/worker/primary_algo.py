# Backward compatibility adapters:
from __future__ import annotations

from typing import TypeVar, overload

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers import (
    InsistSuggest,
    MultiTaskWrapper,
    SpaceTransform,
)
from orion.core.worker.algo_wrappers.space_transform import (  # noqa
    SpaceTransform as SpaceTransformAlgoWrapper,
)
from orion.core.worker.knowledge_base import KnowledgeBase

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)


@overload
def create_algo(
    algo_type: type[AlgoType],
    space: Space,
    knowledge_base: KnowledgeBase,
    **algo_kwargs,
) -> MultiTaskWrapper[InsistSuggest[SpaceTransform[AlgoType]]]:
    ...


@overload
def create_algo(
    algo_type: type[AlgoType],
    space: Space,
    knowledge_base: None = None,
    **algo_kwargs,
) -> InsistSuggest[SpaceTransform[AlgoType]]:
    ...


def create_algo(
    algo_type: type[AlgoType],
    space: Space,
    knowledge_base: KnowledgeBase | None = None,
    **algo_kwargs,
):
    """Create an AlgoWrapper from the given algorithm and wrappers."""

    spaces = [space]
    # Create the spaces for each level, from the top down.
    if knowledge_base:
        space = MultiTaskWrapper.transform_space(space, knowledge_base=knowledge_base)
        spaces.append(space)

    space = InsistSuggest.transform_space(space)
    spaces.append(space)

    space = SpaceTransform.transform_space(space, algo_type=algo_type)
    spaces.append(space)

    assert "algorithm" not in algo_kwargs, (algo_type, algo_kwargs)

    # Create the algo, using the innermost (most transformed) space.
    algo = algo_type(space=spaces.pop(), **algo_kwargs)

    # Create each wrapper, from the bottom up.
    algo = SpaceTransform(space=spaces.pop(), algorithm=algo)
    algo = InsistSuggest(space=spaces.pop(), algorithm=algo)
    if knowledge_base:
        algo = MultiTaskWrapper(space=spaces.pop(), algorithm=algo)
    return algo
