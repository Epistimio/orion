"""Minimal API to run workon"""
import logging
import numbers
from typing import Dict, List, Optional

from orion.core.worker.trial import TrialCM
from orion.core.worker.trial_pacemaker import TrialPacemaker
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
        self._pacemakers = {}

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

    def suggest(self, pool_size: int = 1, experiment_name=None) -> TrialCM:
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
            trials.append(
                RemoteTrial(**trial, exp_working_dir=self.experiment.working_dir)
            )

        # Currently we only return a single trial
        trial = trials[0]

        # Create pacemaker for them
        self._maintain_reservation(trial)
        return TrialCM(self, trial)

    def _maintain_reservation(self, trial):
        self._pacemakers[trial.id] = TrialPacemaker(trial, self)
        self._pacemakers[trial.id].start()

    def release(self, trial, status="interrupted"):
        """Release a trial after work on it finished"""
        experiment_name = self.experiment_name

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        failed = False
        try:
            self._post(
                "release",
                euid=self.experiment_id,
                experiment_name=experiment_name,
                trial_hash=trial.id,
                status=status,
            )
        except Exception as e:
            failed = True
            raise e
        finally:
            pacemaker = self._pacemakers.pop(trial.id, None)
            if pacemaker is not None:
                pacemaker.stop()
            elif not failed:
                raise RuntimeError(
                    f"Trial {trial.id} had no pacemakers. Was it reserved properly?"
                )

    def observe(
        self,
        trial: RemoteTrial,
        results: List[Dict],
        name: str = "objective",
        experiment_name=None,
    ) -> None:
        """Observe the result of a given trial"""
        experiment_name = experiment_name or self.experiment_name

        if isinstance(results, numbers.Number):
            results = [dict(value=results, name=name, type="objective")]

        if experiment_name is None:
            raise ExperiementIsNotSetup("experiment_name is not set")

        if not isinstance(results, list):
            raise TypeError("Results need to be a float or a list of values")

        self._post(
            "observe",
            euid=self.experiment_id,
            experiment_name=experiment_name,
            trial_id=trial.db_id,
            trial_hash=trial.params_id,
            results=results,
        )

        pacemaker = self._pacemakers.pop(trial.id, None)
        if pacemaker is not None:
            pacemaker.stop()
        else:
            log.warning("Trial pacemaker not found! %s", trial.id)

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

    def _update_heartbeat(self, trial):
        return not self.heartbeat(trial)

    def close(self):
        """Release all the resources reserved by the client"""
        if self._pacemakers:
            raise RuntimeError(
                f"There is still reserved trials: {self._pacemakers.keys()}\n"
                "Release all trials before closing the client, using "
                "client.release(trial)."
            )

    def release_all(self):
        """Release all trials that remains to other client can pick them up"""
        pacemakers = list(self._pacemakers.items())

        for k, v in pacemakers:
            log.warning("Trial was not already released! %s", k)
            self.release(RemoteTrial(None, k, []), "interrupted")
            v.stop()

    def __del__(self):
        self.close()
