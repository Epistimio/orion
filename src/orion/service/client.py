import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__file__)


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


@dataclass
class RemoteTrial:
    db_id: str
    params_id: str
    params: List[Dict[str, Any]]


class ClientREST:
    """Implements the basic REST client for the experiment"""

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

        self.experiment = RemoteExperiment(
            euid=payload.get("euid"),
            name=name,
            space=payload.get("space"),
            version=payload.get("version"),
            mode=payload.get("mode"),
            working_dir=payload.get("working_dir"),
            metadata=payload.get("metadata"),
        )
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

        return trials

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
            trial_id=trial.db_id,
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

from orion.client.experiment import ExperimentClient
from orion.core.worker.experiment import Experiment


class ExperimentClientREST(ExperimentClient):
    @classmethod
    def create_experiment(cls):
        pass

    def __init__(self, experiment, executor=None, heartbeat=None):

        endpoint = None
        token = None
        self.rest = ClientREST(endpoint, token)

        # Do not call super here
        # we want to see what is missing

    def is_broken(self):
        try:
            self.is_done
        except RemoteException as exception:
            if exception == "BrokenExperiment":
                return True
        return False

    #
    # REST API overrides
    #

    def is_done(self):
        return self.rest.is_done()

    def suggest(self, pool_size=0):
        remote_trial = self.rest.suggest(pool_size=pool_size)
        return remote_trial

    def observe(self, trial, results):
        self.rest.observe(trial, results)

    def _update_heardbeat(self, trial):
        return not self.rest.heartbeat(trial)

    def storage(self):
        raise RuntimeError("Access to storage is forbidden")


class Client:
    def __init__(self, instance=None, **config) -> None:
        self.config = config
        self.storage = instance

    @classmethod
    def from_command_line(cls, args):
        cmd_config = get_cmd_config(cmdargs)

        instance = setup_storage(cmd_config["storage"])

        return cls(instance, **cmd_config)

    @staticmethod
    def experiment_class():
        return Experiment

    @staticmethod
    def experiment_client_class():
        return ExperimentClient

    def experiment(self):
        """Create a new experiment"""
