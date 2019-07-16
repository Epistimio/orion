# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.base -- Generic Storage Protocol
====================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a generic protocol to allow Orion to communicate using
   different storage backend

"""

from orion.core.utils import Factory


class BaseStorageProtocol:
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend

    """

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        raise NotImplementedError()

    def update_experiment(self, experiment, where=None, **kwargs):
        """Update a the fields of a given trials

        Parameters
        ----------
        experiment: Experiment
            Experiment object to update

        where: Optional[dict]
            constraint experiment must respect

        kwargs: dict
            a dictionary of fields to update

        """
        raise NotImplementedError()

    def fetch_experiments(self, query):
        """Fetch all experiments that match the query"""
        raise NotImplementedError()

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        raise NotImplementedError()

    def reserve_trial(self, *args, **kwargs):
        """Select a pending trial and reserve it for the worker"""
        raise NotImplementedError()

    def fetch_trials(self, query, *args, **kwargs):
        """Fetch all the trials that match the query"""
        raise NotImplementedError()

    def update_trial(self, trial, where=None, **kwargs):
        """Update the fields of a given trials

        Parameters
        ----------
        trial: Trial
            Trial object to update

        where: Optional[dict]
            constraint trial must respect

        kwargs: dict
            a dictionary of fields to update

        returns the updated trial

        """
        raise NotImplementedError()

    def retrieve_result(self, trial, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the db
        """
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
    MongoDB, track, cometML, MLFLow, etc...

    Examples
    --------
    >>> StorageProtocol('track', uri='file://orion_test.json')
    >>> StorageProtocol('legacy', experiment=...)

    """

    pass


# pylint: disable=too-few-public-methods
class ReadOnlyStorageProtocol(object):
    """Read-only interface from a storage protocol.

    .. seealso::

        :py:class:`orion.core.storage.BaseStorageProtocol`
    """

    __slots__ = ('_storage', )
    valid_attributes = {"fetch_trials", "fetch_experiments"}

    def __init__(self, protocol):
        """Init method, see attributes of :class:`BaseStorageProtocol`."""
        self._storage = protocol

    def __getattr__(self, attr):
        """Get attribute only if valid"""
        if attr not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on ReadOnlyStorageProtocol." % attr)

        return getattr(self._storage, attr)
