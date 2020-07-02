# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy` -- Legacy storage
=============================================

.. module:: legacy
   :platform: Unix
   :synopsis: Old Storage implementation

"""
import datetime
import json
import logging

import orion.core
from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database, OutdatedDatabaseError
import orion.core.utils.backward as backward
from orion.core.utils.exceptions import MissingResultFile
from orion.core.worker.trial import Trial, validate_status
from orion.storage.base import BaseStorageProtocol, FailedUpdate, MissingArguments

log = logging.getLogger(__name__)


def get_database():
    """Return current database

    This is a wrapper around the Database Singleton object to provide
    better error message when it is used without being initialized.

    Raises
    ------
    RuntimeError
        If the underlying database was not initialized prior to calling this function

    Notes
    -----
    To initialize the underlying database you must first call `Database(...)`
    with the appropriate arguments for the chosen backend

    """
    return Database()


def setup_database(config=None):
    """Create the Database instance from a configuration.

    Parameters
    ----------
    config: dict
        Configuration for the database backend. If not defined, global configuration
        is used.

    """
    if config is None:
        # TODO: How could we support orion.core.config.storage.database as well?
        config = orion.core.config.database.to_dict()

    db_opts = config
    dbtype = db_opts.pop('type')

    log.debug("Creating %s database client with args: %s", dbtype, db_opts)
    try:
        Database(of_type=dbtype, **db_opts)
    except ValueError:
        if Database().__class__.__name__.lower() != dbtype.lower():
            raise


class Legacy(BaseStorageProtocol):
    """Legacy protocol, store all experiments and trials inside the Database()

    Parameters
    ----------
    config: Dict
        configuration definition passed from experiment_builder
        to storage factory to legacy constructor.
        See `~orion.io.database.Database` for more details
    setup: bool
        Setup the database (create indexes)

    """

    def __init__(self, database=None, setup=True):
        if database is not None:
            setup_database(database)

        self._db = Database()

        if setup:
            self._setup_db()

    def _setup_db(self):
        """Database index setup"""
        if backward.db_is_outdated(self._db):
            raise OutdatedDatabaseError("The database is outdated. You can upgrade it with the "
                                        "command `orion db upgrade`.")

        self._db.index_information('experiment')
        self._db.ensure_index('experiments',
                              [('name', Database.ASCENDING),
                               ('version', Database.ASCENDING)],
                              unique=True)

        self._db.ensure_index('experiments', 'metadata.datetime')

        self._db.ensure_index('trials', 'experiment')
        self._db.ensure_index('trials', 'status')
        self._db.ensure_index('trials', 'results')
        self._db.ensure_index('trials', 'start_time')
        self._db.ensure_index('trials', [('end_time', Database.DESCENDING)])

    def create_experiment(self, config):
        """See :func:`~orion.storage.BaseStorageProtocol.create_experiment`"""
        return self._db.write('experiments', data=config, query=None)

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.update_experiment`"""
        if experiment is not None and uid is not None:
            assert experiment._id == uid

        if uid is None:
            if experiment is None:
                raise MissingArguments('Either `experiment` or `uid` should be set')

            uid = experiment._id

        if where is None:
            where = dict()

        where['_id'] = uid
        return self._db.write('experiments', data=kwargs, query=where)

    def fetch_experiments(self, query, selection=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_experiments`"""
        return self._db.read('experiments', query, selection)

    def fetch_trials(self, experiment=None, uid=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials`"""
        if experiment is not None and uid is not None:
            assert experiment._id == uid

        if uid is None:
            if experiment is None:
                raise MissingArguments('Either `experiment` or `uid` should be set')

            uid = experiment._id

        return self._fetch_trials(dict(experiment=uid))

    def _fetch_trials(self, query, selection=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials`"""
        def sort_key(item):
            submit_time = item.submit_time
            if submit_time is None:
                return 0
            return submit_time

        trials = Trial.build(self._db.read('trials', query=query, selection=selection))
        trials.sort(key=sort_key)

        return trials

    def register_trial(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.register_trial`"""
        self._db.write('trials', trial.to_dict())
        return trial

    def register_lie(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.register_lie`"""
        return self._db.write('lying_trials', trial.to_dict())

    def retrieve_result(self, trial, results_file=None, **kwargs):
        """Parse the results file that was generated by the trial process.

        Parameters
        ----------
        trial: Trial
            The trial object to be updated

        results_file: str
            the file handle to read the result from

        Returns
        -------
        returns the updated trial object

        Notes
        -----
        This does not update the database!

        """
        if results_file is None:
            return trial

        try:
            results = JSONConverter().parse(results_file.name)
        except json.decoder.JSONDecodeError:
            raise MissingResultFile()

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]

        return trial

    def get_trial(self, trial=None, uid=None):
        """See :func:`~orion.storage.BaseStorageProtocol.get_trial`"""
        if trial is not None and uid is not None:
            assert trial._id == uid

        if uid is None:
            if trial is None:
                raise MissingArguments('Either `trial` or `uid` should be set')

            uid = trial.id

        result = self._db.read('trials', {'_id': uid})
        if not result:
            return None

        return Trial(**result[0])

    def _update_trial(self, trial, where=None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.update_trial`"""
        if where is None:
            where = dict()

        where['_id'] = trial.id
        return self._db.write('trials', data=kwargs, query=where)

    def fetch_lost_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_lost_trials`"""
        heartbeat = orion.core.config.worker.heartbeat
        threshold = datetime.datetime.utcnow() - datetime.timedelta(seconds=heartbeat)
        lte_comparison = {'$lte': threshold}
        query = {
            'experiment': experiment._id,
            'status': 'reserved',
            'heartbeat': lte_comparison
        }

        return self._fetch_trials(query)

    def push_trial_results(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.push_trial_results`"""
        rc = self._update_trial(trial, **trial.to_dict(),
                                where={'_id': trial.id, 'status': 'reserved'})
        if not rc:
            raise FailedUpdate()

        return rc

    def set_trial_status(self, trial, status, heartbeat=None):
        """See :func:`~orion.storage.BaseStorageProtocol.set_trial_status`"""
        if heartbeat is None:
            heartbeat = datetime.datetime.utcnow()

        update = dict(
            status=status,
            heartbeat=heartbeat,
            experiment=trial.experiment
        )

        validate_status(status)

        rc = self._update_trial(trial, **update, where={'status': trial.status, '_id': trial.id})

        if not rc:
            raise FailedUpdate()

        trial.status = status

    def fetch_pending_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_pending_trials`"""
        query = dict(
            experiment=experiment._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
        )
        return self._fetch_trials(query)

    def reserve_trial(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.reserve_trial`"""
        query = dict(
            experiment=experiment._id,
            status={'$in': ['interrupted', 'new', 'suspended']}
        )
        # read and write works on a single document
        now = datetime.datetime.utcnow()
        trial = self._db.read_and_write(
            'trials',
            query=query,
            data=dict(
                status='reserved',
                start_time=now,
                heartbeat=now
            ))

        if trial is None:
            return None

        return Trial(**trial)

    def fetch_noncompleted_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_noncompleted_trials`"""
        query = dict(
            experiment=experiment._id,
            status={'$ne': 'completed'}
        )
        return self._fetch_trials(query)

    def count_completed_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.count_completed_trials`"""
        query = dict(
            experiment=experiment._id,
            status='completed'
        )
        return self._db.count('trials', query)

    def count_broken_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.count_broken_trials`"""
        query = dict(
            experiment=experiment._id,
            status='broken'
        )
        return self._db.count('trials', query)

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""
        return self._update_trial(trial, heartbeat=datetime.datetime.utcnow(), status='reserved')

    def fetch_trials_by_status(self, experiment, status):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials_by_status`"""
        query = dict(
            experiment=experiment._id,
            status=status
        )
        return self._fetch_trials(query)
