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


class BaseStorageProtocol:
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend
    """

    def create_experiment(self):
        raise NotImplementedError()

    def fetch_experiments(self):
        raise NotImplementedError()

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        raise NotImplementedError()

    def reserve_trial(self, score_handle, *args, **kwargs):
        raise NotImplementedError()

    def fetch_trials(self, query):
        raise NotImplementedError()

    # def select_trial(self, *args, **kwargs):
    #     """Select a pending trial to be ran"""
    #     raise NotImplementedError()
    #       fetch_trials()

    # def fetch_completed_trials(self):
    #     """Fetch the latest completed trials of this experiment that were not yet observed"""
    #     raise NotImplementedError()

    # def is_done(self, experiment):
    #     """Check if we have reached the maximum number of trials.
    #     This only takes into account completed trials.
    #     So if trials do not complete this may never return True
    #     """
    #     raise NotImplementedError()

    # def get_trial(self, trial):
    #     """Fetch a trial inside storage using its id"""
    #     raise NotImplementedError()

    def update_trial(self, trial, **kwargs):
        """Try to read the result of the trial and update the trial.results attribute"""
        raise NotImplementedError()

    # def push_completed_trial(self, trial):
    #     """Process a trial and set it as a completed
    #     This also registers some statistics for the experiment (best trial, runtime etc...)
    #     """
    #     raise NotImplementedError()
    #
    # def mark_as_broken(self, trial):
    #     """When a trial fails we set it as broken. It means it will to be re ran again."""
    #     raise NotImplementedError()


# pylint: disable=too-few-public-methods,abstract-method
class StorageProtocol(BaseStorageProtocol, metaclass=Factory):
    """Storage protocol is a generic way of allowing Orion to interface with different storage.
    MongoDB, track, cometML, MLFLow, etc...

    Protocol('track', uri='file://orion_test.json')
    Protocol('legacy', experiment=experiment)
    """

    pass
