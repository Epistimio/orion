# -*- coding: utf-8 -*-
"""
History of past trials
======================

Maintain the history of past trials used by an algorithm.

"""


# pylint:disable=protected-access,too-few-public-methods
class TrialsHistory:
    """Maintain a list of all the last seen trials that are on different dependency paths"""

    def __init__(self):
        """Create empty trials history"""
        self.children = []
        self.ids = set()

    def __contains__(self, trial):
        """Return True if the trial is in the observed history"""
        return trial.id in self.ids

    def update(self, trials):
        """Update the list of children trials

        The children history only keeps children. Current children that are now ancestors of
        the new nodes are discarded from the history. This is because we can rebuild the entire
        history from the current children, therefore we only need to keep those.
        """
        descendents = set(self.children)
        for trial in trials:
            descendents -= set(trial.parents)
            descendents.add(trial.id)

        self.ids |= descendents

        self.children = list(sorted(descendents))
