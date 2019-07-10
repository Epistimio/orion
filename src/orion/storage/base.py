# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.base -- Generic Storage Protocol
=============================================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a generic protocol to allow Orion to communicate using
   different storage backend

"""
from orion.core.utils import Factory

from typing import Optional


class BaseStorageProtocol:
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend
    """

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        raise NotImplementedError()

    def update_experiment(self, experiment: 'Experiment', fields: Optional[dict] = None, **kwargs):
        """Update a the fields of a given trials

        Parameters
        ----------

        :param experiment: Experiment object to update
        :param fields: Optional[dict] a dictionary of fields to update
        :param kwargs: a dictionary of fields to update

        Example
        -------

        >>> query = {'status': 'RUNNING'}
        >>> self.update_experiment(experiment, fields=query)


        >>> self.update_experiment(experiment, status='RUNNING')
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

    def fetch_trials(self, query):
        """Feetch all the trials that match the query"""
        raise NotImplementedError()

    def update_trial(self, trial: 'Trial', fields: Optional[dict] = None, **kwargs) -> 'Trial':
        """Update a the fields of a given trials

        Parameters
        ----------

        :param trial: Trial object to update
        :param fields: Optional[dict] a dictionary of fields to update
        :param kwargs: a dictionary of fields to update

        :return the updated trial

        Example
        -------

        >>> query = {'status': 'RUNNING'}
        >>> self.update_trial(trial, fields=query)


        >>> self.update_trial(trial, status='RUNNING')
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
