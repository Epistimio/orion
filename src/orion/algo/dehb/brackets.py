"""Specialization of SHBracketManager to support duplicated trials"""

import logging

from orion.algo.dehb.logger import remove_loguru

remove_loguru()


from dehb.utils import SHBracketManager as SHBracketManagerImpl

logger = logging.getLogger(__name__)


class SHBracketManager(SHBracketManagerImpl):
    """Override the default implementation to ignore duplicated trials in budget accounting"""

    def __init__(self, n_configs, budgets, bracket_id=None, duplicates=None):
        super().__init__(n_configs, budgets, bracket_id)
        self.duplicates = duplicates

    def complete_job(self, budget):
        """Notifies the bracket that a job for a budget has been completed
        This function must be called when a config for a budget has finished evaluation to inform
        the Bracket Manager that no job needs to be waited for and the next rung can begin for the
        synchronous Successive Halving case.
        """
        assert budget in self.budgets
        _max_configs = self.n_configs[list(self.budgets).index(budget)]

        offset = 0
        if self.duplicates is not None:
            offset = self.duplicates.get(str(budget), 0)

        assert self._sh_bracket[budget] - offset < _max_configs
        self._sh_bracket[budget] += 1
