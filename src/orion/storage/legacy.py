# -*- coding: utf-8 -*-
"""
Legacy storage
==============

Old Storage implementation.

"""
import contextlib
import datetime
import json
import logging
import pickle
import time

import bson

import orion.core
import orion.core.utils.backward as backward
from orion.core.io.database import Database, OutdatedDatabaseError, database_factory
from orion.core.utils.exceptions import MissingResultFile
from orion.core.worker.trial import Trial, validate_status
from orion.storage.base import (
    BaseStorageProtocol,
    FailedUpdate,
    LockAcquisitionTimeout,
    LockedAlgorithmState,
    MissingArguments,
    get_uid,
)

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
    return database_factory.create()


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
    dbtype = db_opts.pop("type")

    log.debug("Creating %s database client with args: %s", dbtype, db_opts)
    return database_factory.create(dbtype, **db_opts)


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

        self._db = database_factory.create()

        if setup:
            self._setup_db()

    def _setup_db(self):
        """Database index setup"""
        if backward.db_is_outdated(self._db):
            raise OutdatedDatabaseError(
                "The database is outdated. You can upgrade it with the "
                "command `orion db upgrade`."
            )

        self._db.index_information("experiment")
        self._db.ensure_index(
            "experiments",
            [("name", Database.ASCENDING), ("version", Database.ASCENDING)],
            unique=True,
        )

        self._db.ensure_index("experiments", "metadata.datetime")

        self._db.ensure_index("benchmarks", "name", unique=True)

        self._db.ensure_index("trials", "experiment")
        self._db.ensure_index("trials", "status")
        self._db.ensure_index("trials", "results")
        self._db.ensure_index("trials", "start_time")
        self._db.ensure_index("trials", [("end_time", Database.DESCENDING)])

        self._db.ensure_index("algo", "experiment")

    def create_benchmark(self, config):
        """Insert a new benchmark inside the database"""
        return self._db.write("benchmarks", data=config, query=None)

    def fetch_benchmark(self, query, selection=None):
        """Fetch all benchmarks that match the query"""
        return self._db.read("benchmarks", query, selection)

    def create_experiment(self, config):
        """See :func:`orion.storage.base.BaseStorageProtocol.create_experiment`"""
        exp_rval = self._db.write("experiments", data=config, query=None)
        self.initialize_algorithm_lock(
            experiment_id=config["_id"], algorithm_config=config.get("algorithms", {})
        )
        return exp_rval

    def delete_experiment(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.delete_experiment`"""
        uid = get_uid(experiment, uid)
        return self._db.remove("experiments", query={"_id": uid})

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """See :func:`orion.storage.base.BaseStorageProtocol.update_experiment`"""
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        if uid is not None:
            where["_id"] = uid
        return self._db.write("experiments", data=kwargs, query=where)

    def fetch_experiments(self, query, selection=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_experiments`"""
        return self._db.read("experiments", query, selection)

    def fetch_trials(self, experiment=None, uid=None, where=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_trials`"""
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        where["experiment"] = uid

        return self._fetch_trials(where)

    def _fetch_trials(self, query, selection=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_trials`"""

        def sort_key(item):
            submit_time = item.submit_time
            if submit_time is None:
                return 0
            return submit_time

        trials = Trial.build(self._db.read("trials", query=query, selection=selection))
        trials.sort(key=sort_key)

        return trials

    def register_trial(self, trial):
        """See :func:`orion.storage.base.BaseStorageProtocol.register_trial`"""
        self._db.write("trials", trial.to_dict())
        return trial

    def delete_trials(self, experiment=None, uid=None, where=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.delete_trials`"""
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        if uid is not None:
            where["experiment"] = uid
        return self._db.remove("trials", query=where)

    def retrieve_result(self, trial, **kwargs):
        """Do nothing for the legacy backend.

        Trial object should already have its results at this point.

        Parameters
        ----------
        trial: Trial
            The trial object to be updated

        Returns
        -------
        returns the trial object

        Notes
        -----
        This does not update the database!

        """
        return trial

    def get_trial(self, trial=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.get_trial`"""
        if trial is not None and uid is not None:
            assert trial._id == uid

        if uid is None:
            if trial is None:
                raise MissingArguments("Either `trial` or `uid` should be set")

            uid = trial.id

        result = self._db.read("trials", {"_id": uid})
        if not result:
            return None

        return Trial(**result[0])

    def update_trials(self, experiment=None, uid=None, where=None, **kwargs):
        """See :func:`orion.storage.base.BaseStorageProtocol.update_trials`"""
        uid = get_uid(experiment, uid)
        if where is None:
            where = dict()

        where["experiment"] = uid
        return self._db.write("trials", data=kwargs, query=where)

    def update_trial(self, trial=None, uid=None, where=None, **kwargs):
        """See :func:`orion.storage.base.BaseStorageProtocol.update_trial`"""
        uid = get_uid(trial, uid)

        if where is None:
            where = dict()

        where["_id"] = uid
        return self._db.write("trials", data=kwargs, query=where)

    def fetch_lost_trials(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_lost_trials`"""
        heartbeat = orion.core.config.worker.heartbeat
        threshold = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=heartbeat * 5
        )
        lte_comparison = {"$lte": threshold}
        query = {
            "experiment": experiment._id,
            "status": "reserved",
            "heartbeat": lte_comparison,
        }

        return self._fetch_trials(query)

    def push_trial_results(self, trial):
        """See :func:`orion.storage.base.BaseStorageProtocol.push_trial_results`"""
        rc = self.update_trial(
            trial, **trial.to_dict(), where={"_id": trial.id, "status": "reserved"}
        )
        if not rc:
            raise FailedUpdate()

        return rc

    def set_trial_status(self, trial, status, heartbeat=None, was=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.set_trial_status`"""
        heartbeat = heartbeat or datetime.datetime.utcnow()
        was = was or trial.status

        validate_status(status)
        validate_status(was)

        update = dict(status=status, heartbeat=heartbeat, experiment=trial.experiment)

        rc = self.update_trial(trial, **update, where={"status": was, "_id": trial.id})

        if not rc:
            raise FailedUpdate()

        trial.status = status

    def fetch_pending_trials(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_pending_trials`"""
        query = dict(
            experiment=experiment._id,
            status={"$in": ["new", "suspended", "interrupted"]},
        )
        return self._fetch_trials(query)

    def reserve_trial(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.reserve_trial`"""
        query = dict(
            experiment=experiment._id,
            status={"$in": ["interrupted", "new", "suspended"]},
        )
        # read and write works on a single document
        now = datetime.datetime.utcnow()
        trial = self._db.read_and_write(
            "trials",
            query=query,
            data=dict(status="reserved", start_time=now, heartbeat=now),
        )

        if trial is None:
            return None

        return Trial(**trial)

    def fetch_noncompleted_trials(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_noncompleted_trials`"""
        query = dict(experiment=experiment._id, status={"$ne": "completed"})
        return self._fetch_trials(query)

    def count_completed_trials(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.count_completed_trials`"""
        query = dict(experiment=experiment._id, status="completed")
        return self._db.count("trials", query)

    def count_broken_trials(self, experiment):
        """See :func:`orion.storage.base.BaseStorageProtocol.count_broken_trials`"""
        query = dict(experiment=experiment._id, status="broken")
        return self._db.count("trials", query)

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""
        return self.update_trial(
            trial, heartbeat=datetime.datetime.utcnow(), where={"status": "reserved"}
        )

    def fetch_trials_by_status(self, experiment, status):
        """See :func:`orion.storage.base.BaseStorageProtocol.fetch_trials_by_status`"""
        query = dict(experiment=experiment._id, status=status)
        return self._fetch_trials(query)

    def initialize_algorithm_lock(self, experiment_id, algorithm_config):
        """See :func:`orion.storage.base.BaseStorageProtocol.initialize_algorithm_lock`"""
        return self._db.write(
            "algo",
            {
                "experiment": experiment_id,
                "configuration": algorithm_config,
                "locked": 0,
                "state": None,
                "heartbeat": datetime.datetime.utcnow(),
            },
        )

    def release_algorithm_lock(self, experiment=None, uid=None, new_state=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.release_algorithm_lock`"""
        uid = get_uid(experiment, uid)

        new_data = dict(
            experiment=uid,
            locked=0,
            heartbeat=datetime.datetime.utcnow(),
        )
        if new_state is not None:
            new_data["state"] = pickle.dumps(new_state)

        self._db.read_and_write(
            "algo",
            query=dict(experiment=uid, locked=1),
            data=new_data,
        )

    def get_algorithm_lock_info(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.get_algorithm_lock_info`"""
        uid = get_uid(experiment, uid)
        locks = self._db.read("algo", {"experiment": uid})

        if not locks:
            return None

        algo_state_lock = locks[0]
        return LockedAlgorithmState(
            state=pickle.loads(algo_state_lock["state"])
            if algo_state_lock["state"] is not None
            else None,
            configuration=algo_state_lock["configuration"],
            locked=algo_state_lock["locked"],
        )

    def delete_algorithm_lock(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.delete_algorithm_lock`"""
        uid = get_uid(experiment, uid)
        return self._db.remove("algo", query={"experiment": uid})

    @contextlib.contextmanager
    def acquire_algorithm_lock(
        self, experiment=None, uid=None, timeout=60, retry_interval=1
    ):
        """See :func:`orion.storage.base.BaseStorageProtocol.acquire_algorithm_lock`"""
        uid = get_uid(experiment, uid)

        algo_state_lock = None
        start = time.perf_counter()
        while algo_state_lock is None and time.perf_counter() - start < timeout:
            algo_state_lock = self._db.read_and_write(
                "algo",
                query=dict(experiment=uid, locked=0),
                data=dict(
                    locked=1,
                    heartbeat=datetime.datetime.utcnow(),
                ),
            )
            if algo_state_lock is None:
                time.sleep(retry_interval)

        if algo_state_lock is None:
            raise LockAcquisitionTimeout

        locked_algo_state = LockedAlgorithmState(
            state=pickle.loads(algo_state_lock["state"])
            if algo_state_lock["state"] is not None
            else None,
            configuration=algo_state_lock["configuration"],
            locked=True,
        )

        try:
            yield locked_algo_state
        except Exception:
            # Reset algo to state fetched lock time
            locked_algo_state.reset()
            raise
        finally:
            self.release_algorithm_lock(uid=uid, new_state=locked_algo_state.state)
