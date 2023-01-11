"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
from __future__ import annotations

from logging import getLogger as get_logger

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers.transform_wrapper import AlgoT, TransformWrapper
from orion.core.worker.transformer import (
    ReshapedSpace,
    TransformedSpace,
    build_required_space,
)
from orion.core.worker.trial import Trial

logger = get_logger(__name__)


class SpaceTransform(TransformWrapper[AlgoT]):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    Parameters
    ----------
    space : `orion.algo.space.Space`
       The original definition of a problem's parameters space.
    algorithm: instance of `BaseAlgorithm`
        Algorithm to be wrapped.
    """

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space=space, algorithm=algorithm)

    @property
    def transformed_space(self) -> TransformedSpace | ReshapedSpace:
        """The transformed space (after transformations).
        This is only exposed to the wrapped algo, not to classes outside of this.
        """
        transformed_space = self.algorithm.space
        assert isinstance(transformed_space, (TransformedSpace, ReshapedSpace))
        return transformed_space

    # pylint: disable=arguments-differ
    @classmethod
    def transform_space(
        cls, space: Space, algo_type: type[BaseAlgorithm]
    ) -> TransformedSpace | ReshapedSpace:
        """Transform the space, so that the algorithm that is passed to the constructor already
        has the right space.
        """
        transformed_space = build_required_space(
            space,
            type_requirement=algo_type.requires_type,
            shape_requirement=algo_type.requires_shape,
            dist_requirement=algo_type.requires_dist,
        )
        assert isinstance(transformed_space, (TransformedSpace, ReshapedSpace))
        return transformed_space

    def transform(self, trial: Trial) -> Trial:
        self._verify_trial(trial)
        return self.transformed_space.transform(trial)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self.transformed_space.reverse(trial)

    def _verify_trial(self, trial: Trial, space: Space | None = None) -> None:
        space = space or self.space
        space.assert_contains(trial)
