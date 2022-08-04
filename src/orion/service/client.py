import logging
from typing import Any, Dict, List

import requests

from orion.core.worker.trial import Trial

log = logging.getLogger(__file__)


class ExperiementIsNotSetup(Exception):
    pass


class ClientREST:
    """Implements the basic REST client for the experiment"""

    def __init__(self, endpoint, token) -> None:
        self.endpoint = endpoint
        self.token = token
        self.experiment_name = None

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
        raise RuntimeError(f"Remote server returned error code {status}: {error}")

    def new_experiment(self, name, **config) -> str:
        self._post("experiment", name=name, **config)

        self.experiment_name = name
        return self.experiment_name

    def suggest(self, pool_size: int = 1, experiment_name=None) -> List[Trial]:
        experiment_name = experiment_name or self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        log.debug("client: suggest: %s", experiment_name)
        result = self._post(
            "suggest", experiment_name=experiment_name, pool_size=pool_size
        )
        return result["trials"]

    def observe(self, trial: Trial, results: List[Dict], experiment_name=None) -> None:
        experiment_name = experiment_name or self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        self._post(
            "observe",
            experiment_name=experiment_name,
            trial_id=trial["_id"],
            results=results,
        )

    def is_done(self) -> bool:
        return self._post("is_done", experiment_name=self.experiment_name)

    def heartbeat(self, trial: Trial) -> None:
        self._post("heartbeat", trial_id=trial["_id"])


# WIP

from orion.client.experiment import ExperimentClient


class ExperimentClientREST(ExperimentClient):
    def __init__(self, endpoint, token):
        self.rest = ClientREST(endpoint, token)

        # Do not call super here
        # we want to see what is missing
