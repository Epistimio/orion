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
        self.parents = []

    def update_parents(self, trials):
        """
        Update the current parents by discarding any old ones while keeping the ones that don't
        have children
        """
        for trial in trials:
            self.parents = [parent for parent in self.parents if parent not in trial.parents]
        self.parents.extend([trial._id for trial in trials])
