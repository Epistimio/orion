"""Minimal API to run workon"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from bson import ObjectId

from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import BrokenExperiment, CompletedExperiment
from orion.core.utils.flatten import unflatten
from orion.core.worker.trial import AlreadyReleased

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
    _params: List[Dict[str, Any]]

    # Populated by the worker locally
    exp_working_dir: Optional[str] = None
    working_dir: Optional[str] = None

    # This is used to copy the parent
    parent = None

    @staticmethod
    def make_json_from_trial(trial) -> dict:
        """Encode a trial into a json Remote Trial"""
        if trial is None:
            return None

        # We only send the minimal amount of information to the client
        # to force the client the communicate with us
        # to avoid having invisible issues.
        data = trial.to_dict()

        small_trial = {}
        small_trial["db_id"] = str(data["_id"])
        small_trial["params_id"] = str(data["id"])
        small_trial["_params"] = data["params"]

        try:
            small_trial["working_dir"] = str(trial.get_working_dir())
        except RuntimeError:
            small_trial["working_dir"] = None

        return small_trial

    # Trial API
    # =========

    def to_dict(self):
        """Convert a remote trial into a json friendly dictionary"""
        return dict(_id=ObjectId(self.db_id), id=self.params_id, params=self._params)

    @property
    def id(self):
        """Returns the parameter hash"""
        return self.params_id

    @property
    def params(self):
        """Parameters of the trial"""
        return unflatten({param["name"]: param["value"] for param in self._params})

    @property
    def status(self):
        """Returns the status of the trial, Remote trials are always reserved"""
        return "reserved"


ALLOWED_EXCEPTION = {
    "orion.core.io.database.DuplicateKeyError": DuplicateKeyError,
    "orion.core.worker.trial.AlreadyReleased": AlreadyReleased,
    "orion.core.utils.exceptions.CompletedExperiment": CompletedExperiment,
    "orion.core.utils.exceptions.BrokenExperiment": BrokenExperiment,
    "builtins.ValueError": ValueError,
}


def raise_remote_exception(typename, args):
    """Reconstruct the remote exception if it is from a list of known exceptions,
    else throw a generic RemoteException.


    """
    exectype = ALLOWED_EXCEPTION.get(typename)

    if exectype is None:
        if len(args) == 1:
            args[0] = typename + ": " + args[0]

        raise RemoteException(*args)

    raise exectype(*args)


# pylint: disable=too-few-public-methods
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

        return raise_remote_exception(**payload.pop("exception"))
