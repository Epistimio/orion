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

        result = requests.post(self.endpoint + "/" + path, data=data)

        if result.status_code >= 200 and result.status_code < 300:
            payload = result.json()
            status = payload.pop("status")

            if status == 0:
                return payload

            raise RuntimeError(f"Remote server returned {status}")

        raise RuntimeError(f"{result}")

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
