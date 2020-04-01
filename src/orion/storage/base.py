# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.base` -- Generic Storage Protocol
=====================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a generic protocol to allow Orion to communicate using
              different storage backend

"""

import logging

import orion.core
from orion.core.utils import (AbstractSingletonType, SingletonFactory)


log = logging.getLogger(__name__)


class FailedUpdate(Exception):
    """Exception raised when we are unable to update a trial' status"""

    pass


class MissingArguments(Exception):
    """Raised when calling a function without the minimal set of parameters"""

    pass


class BaseStorageProtocol(metaclass=AbstractSingletonType):
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend

    """

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        raise NotImplementedError()

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """Update a the fields of a given trials

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint experiment must respect

        **kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match

        """
        raise NotImplementedError()

    def fetch_experiments(self, query, selection=None):
        """Fetch all experiments that match the query"""
        raise NotImplementedError()

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        raise NotImplementedError()

    def register_lie(self, trial):
        """Register a *fake* trial created by the strategist.

        The main difference between fake trial and orignal ones is the addition of a fake objective
        result, and status being set to completed. The id of the fake trial is different than the id
        of the original trial, but the original id can be computed using the hashcode on parameters
        of the fake trial. See mod:`orion.core.worker.strategy` for more information and the
        Strategist object and generation of fake trials.

        Parameters
        ----------
        trial: `Trial` object
            Fake trial to register in the database

        """
        raise NotImplementedError()

    def reserve_trial(self, experiment):
        """Select a pending trial and reserve it for the worker

        Returns
        -------
        Returns the reserved trial or None if no trials were found

        """
        raise NotImplementedError()

    def fetch_trials(self, experiment=None, uid=None):
        """Fetch all the trials of an experiment in the database

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        Returns
        -------
        return none if the experiment is not found,

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match

        """
        raise NotImplementedError()

    def get_trial(self, trial=None, uid=None):
        """Fetch a single trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to retrieve from the database

        uid: str, optional
            trial id used to retrieve the trial object

        Returns
        -------
        return none if the trial is not found,

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        raise NotImplementedError()

    def fetch_lost_trials(self, experiment):
        """Fetch all trials that have a heartbeat older than
        some given time delta (2 minutes by default)
        """
        raise NotImplementedError()

    def retrieve_result(self, trial, *args, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the trial object
        """
        raise NotImplementedError()

    def push_trial_results(self, trial):
        """Push the trial's results to the database"""
        raise NotImplementedError()

    def set_trial_status(self, trial, status, heartbeat=None):
        """Update the trial status and the heartbeat

        Raises
        ------
        FailedUpdate
            The exception is raised if the status of the trial object
            does not match the status in the database

        """
        raise NotImplementedError()

    def fetch_pending_trials(self, experiment):
        """Fetch all trials that are available to be executed by a worker,
        this includes new, suspended and interrupted trials
        """
        raise NotImplementedError()

    def fetch_noncompleted_trials(self, experiment):
        """Fetch all non completed trials"""
        raise NotImplementedError()

    def fetch_trials_by_status(self, experiment, status):
        """Fetch all trials with the given status"""
        raise NotImplementedError()

    def count_completed_trials(self, experiment):
        """Count the number of completed trials"""
        raise NotImplementedError()

    def count_broken_trials(self, experiment):
        """Count the number of broken trials"""
        raise NotImplementedError()

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class Storage(BaseStorageProtocol, metaclass=SingletonFactory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
    MongoDB, track, cometML, MLFLow, etc...

    Examples
    --------
    >>> Storage('track', uri='file://orion_test.json')
    >>> Storage('legacy', experiment=...)

    Notes
    -----
    When retrieving an already initialized Storage object you should use `get_storage`.
    `Storage()` should only be used for initialization purposes as `get_storage`
    raises more granular error messages.

    """

    pass


def get_storage():
    """Return current storage

    This is a wrapper around the Storage Singleton object to provide
    better error message when it is used without being initialized.

    Raises
    ------
    RuntimeError
        If the underlying storage was not initialized prior to calling this function

    Notes
    -----
    To initialize the underlying storage you must first call `Storage(...)`
    with the appropriate arguments for the chosen backend

    """
    return Storage()


def setup_storage(storage=None, debug=False):
    """Create the storage instance from a configuration.

    Parameters
    ----------
    config: dict, optional
        Configuration for the storage backend. If not defined, global configuration
        is used.
    debug: bool, optional
        If using in debug mode, the storage config is overrided with legacy:EphemeralDB.
        Defaults to False.

    """
    if storage is None:
        storage = orion.core.config.storage.to_dict()

    if storage.get('type') == 'legacy' and 'database' not in storage:
        storage['database'] = orion.core.config.storage.database.to_dict()
    elif storage.get('type') is None and 'database' in storage:
        storage['type'] = 'legacy'

    if debug:
        storage = {'type': 'legacy', 'database': {'type': 'EphemeralDB'}}

    storage_type = storage.pop('type')

    log.debug("Creating %s storage client with args: %s", storage_type, storage)
    try:
        Storage(of_type=storage_type, **storage)
    except ValueError:
        if Storage().__class__.__name__.lower() != storage_type.lower():
            raise


# pylint: disable=too-few-public-methods
class ReadOnlyStorageProtocol(object):
    """Read-only interface from a storage protocol.

    .. seealso::

        :py:class:`orion.core.storage.BaseStorageProtocol`
    """

    __slots__ = ('_storage', )
    valid_attributes = {
        'get_trial',
        'fetch_trials',
        'fetch_experiments',
        'count_broken_trials',
        'count_completed_trials',
        'fetch_noncompleted_trials',
        'fetch_pending_trials',
        'fetch_lost_trials',
        'fetch_trials_by_status'
    }

    def __init__(self, protocol):
        """Init method, see attributes of :class:`BaseStorageProtocol`."""
        self._storage = protocol

    def __getattr__(self, attr):
        """Get attribute only if valid"""
        if attr not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on ReadOnlyStorageProtocol." % attr)

        return getattr(self._storage, attr)
