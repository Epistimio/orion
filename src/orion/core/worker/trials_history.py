# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.trials_history` -- History of past trials
=================================================================

.. module:: trials_history
    :platform: Unix
    :synopsis: Maintain the history of past trials used by an algorithm

"""


# pylint:disable=protected-access,too-few-public-methods
class TrialsHistory:
    """Maintain a list of all the last seen trials that are on different dependency paths"""

    def __init__(self):
        """Create empty trials history"""
        self.children = []

    def update(self, trials):
        """Update the list of children trials

        This history only keeps children. Current children that are now ancestors of
        the new nodes are discarded from the history. This is because we can rebuild the entire
        history from the current children, therefore we only need to keep those.
        """
        descendents = set(self.children)
        for trial in trials:
            descendents -= set(trial.parents)
            descendents.add(trial._id)

        self.children = list(sorted(descendents))
