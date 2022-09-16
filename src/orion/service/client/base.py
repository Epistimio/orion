"""Minimal API to run workon"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class ExperiementIsNotSetup(Exception):
    """Raised when the API is missing an experiment name"""


class RemoteException(Exception):
    """Raised from a request that failed"""


@dataclass
class RemoteExperiment:
    """Simple read only experiment"""

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
    """Simple read only trial"""

    db_id: str
    params_id: str
    params: List[Dict[str, Any]]

    # Populated by the worker locally
    exp_working_dir: Optional[str] = None
    working_dir: Optional[str] = None

    # This is used to copy the parent
    parent = None


class BaseClientREST:
    """Standard handling of REST requests for orion endpoints"""

    def __init__(self, endpoint, token) -> None:
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
