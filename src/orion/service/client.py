import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from orion.storage.base import setup_storage

log = logging.getLogger(__name__)


class ExperiementIsNotSetup(Exception):
    pass


class RemoteException(Exception):
    pass


@dataclass
class RemoteExperiment:
    euid: str
    name: str
    space: dict
    version: str
    mode: str
    working_dir: str
    metadata: dict
    max_trials: int


@dataclass
class RemoteTrial:
    db_id: str
    params_id: str
    params: List[Dict[str, Any]]

    # Populated by the worker locally
    exp_working_dir: Optional[str] = None
    working_dir: Optional[str] = None


class ClientREST:
    """Implements the basic REST client for the experiment

    Its goal is limited to communicating to the remote algo,
    For generic query you should implement the functionality inside the rest storage.

    """

    def __init__(self, endpoint, token) -> None:
        self.endpoint = endpoint
        self.token = token
        self.experiment = None

    @property
    def experiment_name(self) -> Optional[str]:
        """returns the current experiment name"""
        return self.experiment.name if self.experiment else None

    @property
    def experiment_id(self) -> Optional[str]:
        """returns the current experiment id"""
        return self.experiment.euid if self.experiment else None

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

    def new_experiment(self, name, **config) -> RemoteExperiment:
        """Create a new experiment"""
        payload = self._post("experiment", name=name, **config)

        payload.pop("experiment_name", None)
        payload["name"] = name
        self.experiment = RemoteExperiment(**payload)
        return self.experiment

    def suggest(self, pool_size: int = 1, experiment_name=None) -> List[RemoteTrial]:
        """Generate a new trial for the current experiment"""
        experiment_name = experiment_name or self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        log.debug("client: suggest: %s", experiment_name)
        result = self._post(
            "suggest", experiment_name=experiment_name, pool_size=pool_size
        )

        trials = []
        for trial in result["trials"]:
            trials.append(RemoteTrial(**trial))

        # Currently we only return a single trial
        return trials[0]

    def observe(
        self, trial: RemoteTrial, results: List[Dict], experiment_name=None
    ) -> None:
        """Observe the result of a given trial"""
        experiment_name = experiment_name or self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        self._post(
            "observe",
            euid=self.experiment_id,
            experiment_name=experiment_name,
            trial_id=trial.db_id,
            trial_hash=trial.params_id,
            results=results,
        )

    def is_done(self) -> bool:
        """returns true if the experiment is done"""
        payload = self._post(
            "is_done", experiment_name=self.experiment_name, euid=self.experiment_id
        )
        return payload.get("is_done", True)

    def heartbeat(self, trial: RemoteTrial) -> bool:
        """Update the heartbeat of a given trial, returns true if the heartbeat was updated.
        if not this means the trial was not running and we should probably stop the pacemaker
        """
        payload = self._post("heartbeat", trial_id=trial.db_id)
        return payload["updated"]


# WIP

import orion.core
from orion.client.experiment import ExperimentClient
from orion.plotting.base import PlotAccessor


class ExperimentClientREST(ExperimentClient):
    """REST Client for an experiment

    Notes
    -----
    The main difference with the REST client is that the algorithm is not running alongside the client;
    instead it is ran on the server.

    So on this client there are no (Trial) Producer, instead it relies on the rest API to suggest
    the trials.

    The client is composed of two mains objects; the REST client which is in charge of communicating with the
    algo running remotely (suggest & observe); and the REST storage which is in charge of fetching information.

    The REST storage is mostly read only as we should not modify the experiment state while it is running.

    To achieve this, the REST client overrides some selected methods from the ExperimentClient which short-circuits
    the execution of the algorithm to use the rest calls.

    """

    @staticmethod
    def create_experiment(
        name,
        version=None,
        space=None,
        algorithms=None,
        strategy=None,
        max_trials=None,
        max_broken=None,
        storage=None,
        branching=None,
        max_idle_time=None,
        heartbeat=None,
        working_dir=None,
        debug=False,
        executor=None,
    ):
        """Instantiate an experiment using the REST API instead of relying on local storage"""

        endpoint, token = storage["endpoint"], storage["token"]

        rest = ClientREST(endpoint, token)
        storage_instance = setup_storage(storage)

        experiment = rest.new_experiment(
            name,
            version=version,
            space=space,
            algorithms=algorithms,
            strategy=strategy,
            max_trials=max_trials,
            max_broken=max_broken,
            storage=storage,
            branching=branching,
            max_idle_time=max_idle_time,
            heartbeat=heartbeat,
            working_dir=working_dir,
            debug=debug,
        )

        client = ExperimentClientREST(
            experiment,
            executor=executor,
            heartbeat=heartbeat,
        )
        client.storage = storage_instance
        client.rest = rest
        return client

    def __init__(self, experiment, executor=None, heartbeat=None):
        # Do not call super here; we do not want to instantiate the producer
        self.rest = None
        self.storage = None

        if heartbeat is None:
            heartbeat = orion.core.config.worker.heartbeat

        self._experiment = experiment
        self.heartbeat = heartbeat
        self._executor = executor
        self._executor_owner = False

        self.plot = PlotAccessor(self)

    def release(self, trial, status):
        pass

    #
    # REST API overrides
    #

    @property
    def is_broken(self):
        """See `~ExperimentClient.is_broken`"""
        try:
            self.is_done
        except RemoteException as exception:
            if exception == "BrokenExperiment":
                return True
        return False

    @property
    def is_done(self):
        """See `~ExperimentClient.is_done`"""
        return self.rest.is_done()

    def suggest(self, pool_size=0):
        """See `~ExperimentClient.suggest`"""
        remote_trial = self.rest.suggest(pool_size=pool_size)
        return remote_trial

    def observe(self, trial, results):
        """See `~ExperimentClient.observe`"""
        self.rest.observe(trial, results)

    def _update_heardbeat(self, trial):
        return not self.rest.heartbeat(trial)

    def storage(self):
        """See `~ExperimentClient.storage`"""
        raise RuntimeError("Access to storage is forbidden")
