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

    def __init__(self, uri=None):
        """INIT METHOD"""
        self.converter = JSONConverter()
        self._db = Database()
        self._setup_db()

    def _setup_db(self):
        self._db.ensure_index('experiments',
                              [('name', Database.ASCENDING),
                               ('metadata.user', Database.ASCENDING)],
                              unique=True)
        self._db.ensure_index('experiments', 'metadata.datetime')

        self._db.ensure_index('trials', 'experiment')
        self._db.ensure_index('trials', 'status')
        self._db.ensure_index('trials', 'results')
        self._db.ensure_index('trials', 'start_time')
        self._db.ensure_index('trials', [('end_time', Database.DESCENDING)])

    def create_experiment(self):
        raise NotImplementedError()

    def fetch_experiments(self, query):
        return self._db.read('experiments', query)

    def fetch_trials(self, query, selection=None):
        return [Trial(**t) for t in self._db.read('trials', query=query)]

    def register_trial(self, trial):
        """Legacy function @see register_trial"""
        self._db.write('trials', trial.to_dict())
        return trial

    def retrieve_result(self, trial, results_file=None, **kwargs):
        """Read the results from the trial and append it to the trial object"""
        results = self.converter.parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]
        return trial

    def update_trial(self, trial, **kwargs):
        self._db.write('trials', trial.to_dict(), query={'_id': trial.id})
