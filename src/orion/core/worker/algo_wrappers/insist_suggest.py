""" Defines a wrapper that makes `suggest` more likely to return a trial (by asking a few times).
"""
from __future__ import annotations

from logging import getLogger as get_logger

from orion.algo.space import Space
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoT, AlgoWrapper
from orion.core.worker.trial import Trial

logger = get_logger(__name__)


class InsistSuggest(AlgoWrapper[AlgoT]):
    """Wrapper that calls `suggest` a few times until the wrapped algo produces a new trial."""

    def __init__(self, space: Space, algorithm: AlgoT):
        super().__init__(space=space, algorithm=algorithm)
        self.max_suggest_attempts = 100

    def suggest(self, num: int = 1) -> list[Trial]:
        trials: list[Trial] = []

        for suggest_attempt in range(1, self.max_suggest_attempts + 1):
            trials = super().suggest(num)
            if trials:
                if suggest_attempt > 1:
                    logger.debug(
                        f"Succeeded in suggesting new trials after {suggest_attempt} attempts."
                    )
                return trials

            if self.is_done:
                logger.debug(
                    f"Algorithm is done! (after {suggest_attempt} sampling attempts)."
                )
                break

        logger.warning(
            f"Unable to sample a new trial from the algorithm, even after "
            f"{self.max_suggest_attempts} attempts! Returning an empty list."
        )
        return []
