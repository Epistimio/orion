# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy` -- Legacy storage
=============================================================================

.. module:: legacy
   :platform: Unix
   :synopsis: Old Storage implementation

"""
import datetime
import random

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
        self._last_fetched = None

    def register_trial(self, trial):
        """Legacy function @see create_trial"""

        stamp = datetime.datetime.utcnow()
        trial.experiment = self.experiment._id
        trial.status = 'new'
        trial.submit_time = stamp

        Database().write('trials', trial.to_dict())
        return trial

    def reserve_trial(self, score_handle, *args, **kwargs):
        """Legacy function mark a trial as reserved since it will be ran shortly"""
        if score_handle is not None and not callable(score_handle):
            raise ValueError("Argument `score_handle` must be callable with a `Trial`.")

        query = dict(
            experiment=self.experiment._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
        )
        new_trials = self.fetch_trials(query)

        if not new_trials:
            return None

        selected_trial = random.sample(new_trials, 1)[0]

        # Query on status to ensure atomicity. If another process change the
        # status meanwhile, read_and_write will fail, because query will fail.
        query = {'_id': selected_trial.id, 'status': selected_trial.status}

        update = dict(status='reserved')

        if selected_trial.status == 'new':
            update["start_time"] = datetime.datetime.utcnow()

        selected_trial_dict = Database().read_and_write(
            'trials', query=query, data=update)

        if selected_trial_dict is None:
            selected_trial = self.reserve_trial(score_handle=score_handle)
        else:
            selected_trial = Trial(**selected_trial_dict)

        return selected_trial

    def fetch_completed_trials(self):
        """Fetch all the trials that are marked as completed"""

        query = dict(
            status='completed',
            end_time={'$gte': self._last_fetched}
        )
        completed_trials = self.fetch_trials(query)
        self._last_fetched = datetime.datetime.utcnow()

        return completed_trials

    # def is_done(self, experiment):
    #     """Check if we have reached the maximum number of completed trials"""
    #     return self.experiment.is_done

    def push_completed_trial(self, trial):
        """Make the trial as complete and update experiment statistics"""
        trial.end_time = datetime.datetime.utcnow()
        trial.status = 'completed'
        Database().write('trials', trial.to_dict(), query={'_id': trial.id})

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
