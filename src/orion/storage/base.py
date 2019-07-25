# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.base -- Generic Storage Protocol
====================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a generic protocol to allow Orion to communicate using
   different storage backend

"""

from orion.core.utils import (AbstractSingletonType, SingletonFactory)


class BaseStorageProtocol(metaclass=AbstractSingletonType):
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

        **kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        """
        raise NotImplementedError()

    def fetch_experiments(self, query, *args, **kwargs):
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

        Returns
        -------
        returns true if the underlying storage was updated

        """
        raise NotImplementedError()

    def retrieve_result(self, trial, *args, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the trial object
        """
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class Storage(BaseStorageProtocol, metaclass=SingletonFactory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
    MongoDB, track, cometML, MLFLow, etc...

    Examples
    --------
    >>> Storage('track', uri='file://orion_test.json')
    >>> Storage('legacy', experiment=...)

    """

    pass


get_storage = Storage


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
