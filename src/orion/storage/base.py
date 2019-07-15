# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.base -- Generic Storage Protocol
=============================================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a generic protocol to allow Orion to communicate using
   different storage backend

"""
from typing import Optional

from orion.core.utils import Factory


class BaseStorageProtocol:
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend
    """

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        raise NotImplementedError()

    def update_experiment(self, experiment: 'Experiment', fields: Optional[dict] = None,
                          where: Optional[dict] = None, **kwargs):
        """Update a the fields of a given trials

        Parameters
        ----------
        experiment: Experiment object to update
        fields: Optional[dict] a dictionary of fields to update
        where: constraint experiment must respect
        kwargs: a dictionary of fields to update

        Example
        -------
        .. code-block:: python
            query = {'status': 'RUNNING'}
            BaseStorageProtocol.update_experiment(experiment, fields=query)

            BaseStorageProtocol.update_experiment(experiment, status='RUNNING')

        """
        raise NotImplementedError()

    def fetch_experiments(self, query):
        """Fetch all experiments that match the query"""
        raise NotImplementedError()

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        raise NotImplementedError()

    def reserve_trial(self, *args, **kwargs):
        """Select a pending trial and book it for the executor"""
        raise NotImplementedError()

    def fetch_trials(self, query, *args, **kwargs):
        """Feetch all the trials that match the query"""
        raise NotImplementedError()

    def update_trial(self, trial: 'Trial', fields: Optional[dict] = None,
                     where: Optional[dict] = None, **kwargs) -> 'Trial':
        """Update a the fields of a given trials

        Parameters
        ----------
        trial: Trial object to update
        fields: Optional[dict] a dictionary of fields to update
        where: constraint trial must respect
        kwargs: a dictionary of fields to update

        returns the updated trial

        Example
        -------
        .. code-block:: python
            query = {'status': 'RUNNING'}
            BaseStorageProtocol.update_trial(trial, fields=query)

            BaseStorageProtocol.update_trial(trial, status='RUNNING')

        """
        raise NotImplementedError()

    def retrieve_result(self, trial, results_file=None, **kwargs):
        """Read the results from the trial and append it to the trial object"""
        raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
    MongoDB, track, cometML, MLFLow, etc...

    Protocol('track', uri='file://orion_test.json')
    Protocol('legacy', experiment=experiment)
    """

    pass


# pylint: disable=too-few-public-methods
class ReadOnlyStorageProtocol(object):
    """Read-only view on a database.

    .. seealso::

        :py:class:`orion.core.io.database.AbstractDB`
    """

    __slots__ = ('_protocol', )
    valid_attributes = {"fetch_trials", "fetch_experiments"}

    def __init__(self, protocol):
        """Init method, see attributes of :class:`AbstractDB`."""
        self._protocol = protocol

    def __getattr__(self, attr):
        """Get attribute only if valid"""
        if attr not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on view-only experiments." % attr)

        return getattr(self._protocol, attr)
