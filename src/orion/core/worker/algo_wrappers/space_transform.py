"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
from __future__ import annotations

from logging import getLogger as get_logger
from typing import Any, Sequence, TypeVar

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoWrapper
from orion.core.worker.transformer import TransformedSpace, build_required_space
from orion.core.worker.trial import Trial

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)

logger = get_logger(__name__)


# pylint: disable=too-many-public-methods
class SpaceTransform(AlgoWrapper[AlgoType]):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    Parameters
    ----------
    algorithm: instance of `BaseAlgorithm`
        Algorithm to be wrapped.
    space : `orion.algo.space.Space`
       The original definition of a problem's parameters space.
    algorithm_config : dict
       Configuration for the algorithm.

    """

    def __init__(self, space: Space, algorithm: AlgoType):
        super().__init__(space=space, algorithm=algorithm)

    @property
    def original_space(self) -> Space:
        """The original space (before transformations).
        This is exposed to the outside, but not to the wrapped algorithm.
        """
        return self.space

    @property
    def transformed_space(self) -> TransformedSpace:
        """The transformed space (after transformations).
        This is only exposed to the wrapped algo, not to classes outside of this.
        """
        return self.algorithm.space

    @classmethod
    def transform_space(cls, space: Space, algo_type: type[BaseAlgorithm]) -> Space:
        """Transform the space, so that the algorithm that is passed to the constructor already
        has the right space.
        """
        return build_required_space(
            space,
            type_requirement=algo_type.requires_type,
            shape_requirement=algo_type.requires_shape,
            dist_requirement=algo_type.requires_dist,
        )

    def transform(self, trial: Trial) -> Trial:
        return self.transformed_space.transform(trial)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self.transformed_space.reverse(trial)

    @property
    def fidelity_index(self) -> str | None:
        """Compute the index of the space where fidelity is.

        Returns None if there is no fidelity dimension.
        """
        return self.algorithm.fidelity_index
