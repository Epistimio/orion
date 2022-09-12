"""Minimal API to run workon"""
import logging
from typing import Dict, List, Optional

from orion.core.utils.flatten import unflatten
from orion.service.client.base import (
    BaseClientREST,
    ExperiementIsNotSetup,
    RemoteException,
    RemoteExperiment,
    RemoteTrial,
)

log = logging.getLogger(__name__)


class WorkonClientREST(BaseClientREST):
    """Minimal REST API, only implements the required method to run workon.

    Its goal is limited to communicating to the remote algo,
    For generic query you should implement the functionality inside the rest storage.

    """

    def __init__(self, endpoint, token) -> None:
        super().__init__(endpoint, token)
        self.experiment = None

    @property
    def experiment_name(self) -> Optional[str]:
        """returns the current experiment name"""
        return self.experiment.name if self.experiment else None

    @property
    def experiment_id(self) -> Optional[str]:
        """returns the current experiment id"""
        return self.experiment.euid if self.experiment else None

    def new_experiment(self, name, **config) -> RemoteException:
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
            trial["params"] = unflatten(
                {param["name"]: param["value"] for param in trial["params"]}
            )
            trials.append(
                RemoteTrial(**trial, exp_working_dir=self.experiment.working_dir)
            )

        # Currently we only return a single trial
        return trials[0]

    def observe(
        self, trial: RemoteTrial, results: List[Dict], experiment_name=None
    ) -> None:
        """Observe the result of a given trial"""
        experiment_name = experiment_name or self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        assert isinstance(results, list)

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