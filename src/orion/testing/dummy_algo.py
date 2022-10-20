"""A dummy algorithm used for unit tests."""
from __future__ import annotations

from typing import ClassVar, Sequence

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.trial import Trial


class FixedSuggestionAlgo(BaseAlgorithm):
    """A dumb algo that always returns the same trial."""

    requires_type: ClassVar[str | None] = "real"
    requires_shape: ClassVar[str | None] = "flattened"
    requires_dist: ClassVar[str | None] = "linear"

    def __init__(
        self,
        space: Space,
        fixed_suggestion: Trial | None = None,
        seed: int | Sequence[int] | None = None,
    ):
        super().__init__(space)
        self.seed = seed
        self.fixed_suggestion = fixed_suggestion or space.sample(1, seed=seed)[0]
        assert self.fixed_suggestion in space

    def suggest(self, num):
        # NOTE: can't register the trial if it's already here. The fixed suggestion is always "new",
        # but the algorithm actually observes it at some point. Therefore, we don't overwrite what's
        # already in the registry.
        if not self.has_suggested(self.fixed_suggestion):
            self.register(self.fixed_suggestion)
            return [self.fixed_suggestion]
        return []
