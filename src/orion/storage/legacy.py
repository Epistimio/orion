# -*- coding: utf-8 -*-
# pylint:disable=protected-access,too-many-public-methods
"""
:mod:`orion.storage.legacy` -- Legacy storage
=============================================================================

.. module:: legacy
   :platform: Unix
   :synopsis: Old Storage implementation

"""
import warnings

from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database
from orion.core.worker.trial import Trial
from orion.storage.base import BaseStorageProtocol


class Legacy(BaseStorageProtocol):
    """Legacy protocol, forward most request to experiment"""

    def __init__(self, experiment, uri=None):
        """INIT METHOD"""
        self.experiment = experiment
        self.converter = JSONConverter()

    def create_trial(self, trial):
        """Create a trial to be ran in the future"""
        self.experiment.register_trial(trial)

    def register_trial(self, trial):
        """Legacy function @see create_trial"""
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def select_trial(self, *args, **kwargs):
        """Select pending trials that should be ran next"""
        return self.experiment.reserve_trial(*args, **kwargs)

    def reserve_trial(self, *args, **kwargs):
        """Legacy function mark a trial as reserved since it will be ran shortly"""
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def fetch_completed_trials(self):
        """Fetch all the trials that are marked as completed"""
        return self.experiment.fetch_completed_trials()

    def is_done(self, experiment):
        """Check if we have reached the maximum number of completed trials"""
        return self.experiment.is_done

    def push_completed_trial(self, trial):
        """Make the trial as complete and update experiment statistics"""
        self.experiment.push_completed_trial(trial)

    def mark_as_broken(self, trial):
        """Mark the trial as broken to avoid retrying a failing trial"""
        trial.status = 'broken'
        Database().write(
            'trials',
            trial.to_dict(),
            query={
                '_id': trial.id
            }
        )

    def get_stats(self):
        """Return the stats from the experiment"""
        return self.experiment.stats

    def update_trial(self, trial, results_file=None, **kwargs):
        """Read the results from the trial and append it to the trial object"""
        results = self.converter.parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]
        return trial

    def get_trial(self, uid):
        """Fetch the trial from the database"""
        return Trial(**Database().read('trials', {'_id': uid})[0])
