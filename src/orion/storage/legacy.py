# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy` -- Legacy storage
=============================================================================

.. module:: legacy
   :platform: Unix
   :synopsis: Old Storage implementation

"""
from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database
from orion.core.worker.trial import Trial
from orion.core.io.database import DuplicateKeyError
from orion.storage.base import BaseStorageProtocol

from typing import Optional


class Legacy(BaseStorageProtocol):
    """Legacy protocol, forward most request to experiment"""

    def __init__(self, uri=None):
        """INIT METHOD"""
        self.converter = JSONConverter()
        self._db = Database()
        self._setup_db()

    def _setup_db(self):
        """Internal setup"""
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

    def create_experiment(self, config):
        """See :func:`~orion.storage.BaseStorageProtocol.create_experiment`"""
        return self._db.write('experiments', config)

    def update_experiment(self, experiment: 'Experiment', fields: Optional[dict] = None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.update_experiment`"""
        q = {}

        if fields is not None and isinstance(fields, dict):
            q = fields

        elif fields is None:
            q = kwargs

        return self._db.write('experiments', data=q, query={'_id': experiment._id})

    def fetch_experiments(self, query):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_experiments`"""
        return self._db.read('experiments', query)

    def fetch_trials(self, query, selection=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials`"""
        return [Trial(**t) for t in self._db.read('trials', query=query)]

    def register_trial(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.register_trial`"""
        self._db.write('trials', trial.to_dict())
        return trial

    def retrieve_result(self, trial, results_file=None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.retrieve_result`"""
        results = self.converter.parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]
        return trial

    def update_trial(self, trial: Trial, fields: Optional[dict] = None, where: Optional[dict] = None, **kwargs) -> Trial:
        """See :func:`~orion.storage.BaseStorageProtocol.update_trial`"""
        q = {}

        if fields is not None and isinstance(fields, dict):
            q = fields

        elif fields is None:
            q = kwargs

        if where is None:
            where = dict()

        where['_id'] = trial.id
        try:
            was_success = self._db.write('trials', q, query=where)
        except DuplicateKeyError:
            return None

        if not was_success:
            return None

        trials = self.fetch_trials({'_id': trial.id})
        if len(trials) == 1:
            return trials[0]

        return was_success
