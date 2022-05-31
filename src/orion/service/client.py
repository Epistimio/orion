import os
from typing import Any, Dict, List

import requests

from orion.core.worker.trial import Trial


class ClientREST:
    """Implements the basic REST client for the experiment"""

    def __init__(self, endpoint, token) -> None:
        self.endpoint = endpoint
        self.token = token
        self.experiment_id = None

    def _post(self, path: str, **data) -> Any:
        """Basic reply handling, makes sure status is 0, else it will raise an error"""
        data["token"] = self.token

        result = requests.post(self.endpoint + "/" + path, json=data)
        payload = result.json()
        status = payload.pop("status")

        if result.status_code >= 200 and result.status_code < 300:

            if status == 0:
                return payload

        error = payload.pop('error')
        raise RuntimeError(f"Remote server returned error code {status}: {error}")

    def new_experiment(self, **config) -> str:
        result = self._post("experiment", **config)
        self.experiment_id = result["experiment_id"]

        return self.experiment_id

    def suggest(self, count: int = 1) -> List[Trial]:
        result = self._post("suggest", experiment_id=self.experiment_id, count=count)
        return result["trials"]

    def observe(self, trial: Trial, results: List[Dict]) -> None:
        self._post("observe", trial_id=trial.id, results=results)

    def is_done(self) -> bool:
        return self._post("is_done", experiment_id=self.experiment_id)

    def heartbeat(self, trial: Trial) -> None:
        self._post("heartbeat", trial_id=trial.id)


# WIP

from orion.client.experiment import ExperimentClient

class ExperimentClientREST(ExperimentClient):

    def __init__(self, endpoint, token):
        self.rest = ClientREST(endpoint, token)

        # Do not call super here
        # we want to see what is missing