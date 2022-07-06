"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
from __future__ import annotations

from logging import getLogger as get_logger
from typing import TypeVar
import typing

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.algo_wrappers.transform_wrapper import TransformWrapper
from orion.core.worker.transformer import TransformedSpace, build_required_space
from orion.core.worker.trial import Trial

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)
if typing.TYPE_CHECKING:
    from orion.core.worker.warm_start.experiment_config import ExperimentInfo
logger = get_logger(__name__)


class SpaceTransform(TransformWrapper[AlgoType]):
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
        transformed_space = self.algorithm.space
        assert isinstance(transformed_space, TransformedSpace)
        return transformed_space

    # pylint: disable=arguments-differ
    @classmethod
    def transform_space(
        cls, space: Space, algo_type: type[BaseAlgorithm]
    ) -> TransformedSpace:
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
        self._verify_trial(trial)
        return self.transformed_space.transform(trial)

    def reverse_transform(self, trial: Trial) -> Trial:
        return self.transformed_space.reverse(trial)

    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentInfo, list[Trial]]]
    ) -> None:
        super().warm_start(
            [
                (experiment_info, [self.transform(trial) for trial in trials])
                for experiment_info, trials in warm_start_trials
            ]
        )

    def _verify_trial(self, trial: Trial, space: Space | None = None) -> None:
        space = space or self.space
        if trial not in space:
            raise ValueError(
                f"Trial {trial.id} not contained in space:"
                f"\nParams: {trial.params}\nSpace: {space}"
            )
