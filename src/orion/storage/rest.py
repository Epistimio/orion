"""
REST Storage
============

Provide the storae API using orion REST API.

"""
import contextlib
import logging
from typing import Any

import requests

from orion.storage.base import BaseStorageProtocol

log = logging.getLogger(__name__)


class RemoteException(Exception):
    pass


class RESTStorage(BaseStorageProtocol):
    """

    Notes
    -----

    This storage is not a full implementation of the storage protocol.
    It relies on the REST client to handle the missing functionality.

    Parameters
    ----------
    config: Dict
        configuration definition passed from experiment_builder
        to storage factory.

    """

    def __init__(self, config=None, endpoint=None, token=None, **kwargs):
        self.endpoint = endpoint
        self.token = token

    def _post(self, path: str, **data) -> Any:
        """Basic reply handling, makes sure status is 0, else it will raise an error"""
        data["token"] = self.token

        result = requests.post(self.endpoint + "/" + path, json=data)
        payload = result.json()
        log.debug("client: post: %s", payload)
        status = payload.pop("status")

        if result.status_code >= 200 and result.status_code < 300 and status == 0:
            return payload.pop("result")

        error = payload.pop("error")
        raise RemoteException(f"Remote server returned error code {status}: {error}")

    def fetch_benchmark(self, query, selection=None):
        """Fetch all benchmarks that match the query"""
        payload = self._post("fetch_benchmark", query=query, selection=selection)

    def fetch_trials(self, experiment=None, uid=None, where=None):
        """Fetch all the trials of an experiment in the database

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint trials must respect

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
        payload = self._post(
            "fetch_trials", experiment=experiment, uid=uid, where=where
        )

    def get_trial(self, trial=None, uid=None, experiment_uid=None):
        """Fetch a single trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to retrieve from the database

        uid: str, optional
            trial id used to retrieve the trial object

        experiment_uid: str, optional
            experiment id used to retrieve the trial object

        Returns
        -------
        return None if the trial is not found,

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        payload = self._post(
            "get_trial", trial=trial, uid=uid, experiment_uid=experiment_uid
        )

    def fetch_lost_trials(self, experiment):
        """Fetch all trials that have a heartbeat older than
        some given time delta (2 minutes by default)
        """
        payload = self._post("fetch_lost_trials", experiment=experiment)

    def fetch_pending_trials(self, experiment):
        """Fetch all trials that are available to be executed by a worker,
        this includes new, suspended and interrupted trials
        """
        payload = self._post("fetch_pending_trials", experiment=experiment)

    def fetch_noncompleted_trials(self, experiment):
        """Fetch all non completed trials"""
        payload = self._post("fetch_noncompleted_trials", experiment=experiment)

    def fetch_trials_by_status(self, experiment, status):
        """Fetch all trials with the given status"""
        payload = self._post(
            "fetch_trials_by_status", experiment=experiment, status=status
        )

    def count_completed_trials(self, experiment):
        """Count the number of completed trials"""
        payload = self._post("count_completed_trials", experiment=experiment)

    def count_broken_trials(self, experiment):
        """Count the number of broken trials"""
        payload = self._post("count_broken_trials", experiment=experiment)

    #
    # Not Implemented for now
    #   we need to make sure the algo is not using the storage to run

    def create_benchmark(self, config):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def update_trials(self, experiment=None, uid=None, where=None, **kwargs):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def update_trial(
        self, trial=None, uid=None, experiment_uid=None, where=None, **kwargs
    ):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def retrieve_result(self, trial, *args, **kwargs):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def push_trial_results(self, trial):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def set_trial_status(self, trial, status, heartbeat=None, was=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def update_heartbeat(self, trial):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def initialize_algorithm_lock(self, experiment_id, algorithm_config):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def release_algorithm_lock(self, experiment=None, uid=None, new_state=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def get_algorithm_lock_info(self, experiment=None, uid=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def delete_algorithm_lock(self, experiment=None, uid=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    @contextlib.contextmanager
    def acquire_algorithm_lock(self, experiment, timeout=600, retry_interval=1):
        """Not implemented for the REST API"""
        raise NotImplementedError()

    def create_experiment(self, config):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def delete_experiment(self, experiment=None, uid=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def fetch_experiments(self, query, selection=None):
        """Fetch all experiments that match the query"""
        raise NotImplementedError()

    def register_trial(self, trial):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def delete_trials(self, experiment=None, uid=None, where=None):
        """Not Implemented for the rest API"""
        raise NotImplementedError()

    def reserve_trial(self, experiment):
        """Not Implemented for the rest API"""
        raise NotImplementedError()
