"""Specialization of SHBracketManager to support duplicated trials"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy

try:
    from dehb.utils import SHBracketManager as SHBracketManagerImpl

    IMPORT_ERROR = None
except ImportError as exc:
    # pylint: disable=too-few-public-methods
    class SHBracketManagerImpl:
        """Dummy implementation for optional import"""

    IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class SHBracketManager(SHBracketManagerImpl):
    """Override the default implementation to ignore duplicated trials in budget accounting"""

    def __init__(
        self,
        n_configs: numpy.ndarray,
        budgets: numpy.ndarray,
        bracket_id: int,
        duplicates: defaultdict[str, int],
    ):
        super().__init__(n_configs, budgets, bracket_id)
        self.duplicates = duplicates

    def complete_job(self, budget: int) -> None:
        """Notifies the bracket that a job for a budget has been completed
        This function must be called when a config for a budget has finished evaluation to inform
        the Bracket Manager that no job needs to be waited for and the next rung can begin for the
        synchronous Successive Halving case.
        """
        assert budget in self.budgets
        _max_configs: int = self.n_configs[list(self.budgets).index(budget)]

        offset = 0
        if self.duplicates is not None:
            offset = self.duplicates.get(str(budget), 0)

        assert self._sh_bracket[budget] - offset < _max_configs
        self._sh_bracket[budget] += 1
