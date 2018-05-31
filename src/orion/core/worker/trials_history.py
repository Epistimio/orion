# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.trials_history` -- History of past trials
================================================================

.. module:: trials_history
    :platform: Unix
    :synopsis: Maintain the history of past trials used by an algorithm

"""


class TrialsHistory:
    """Maintain a list of all the last seen trials that are on different dependency paths"""

    def __init__(self):
        self.parents = []

    def get_most_recent_parents(self):
        return self.parents

    def update_parents(self, trials):
        for trial in trials:
            self.parents = [parent for parent in self.parents if parent not in trial.parents]
        self.parents.extend([trial._id for trial in trials])
