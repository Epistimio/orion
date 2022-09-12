import datetime
import logging
from typing import Dict

from bson import ObjectId

from orion.core.utils.exceptions import BrokenExperiment
from orion.service.broker.broker import (
    ExperimentBroker,
    RequestContext,
    build_experiment_client,
    get_storage_for_user,
    success,
)

log = logging.getLogger(__name__)


class ObserverError(Exception):
    pass


class LocalExperimentBroker(ExperimentBroker):
    """Creates an experiment client for each request and process the request.
    Parallel requests will instantiate different clients which might generate race conditions,
    locks are required.

    Everything keeps being reinstantiated.

    This is easy to scale because nothing is kept between request,
    we can just load balance the request through n servers.

    """

    def new_experiment(self, request: RequestContext) -> Dict:
        log.debug("Spawning new experiment")

        client = build_experiment_client(request)
        euid = str(client._experiment.id)

        return success(
            dict(
                experiment_name=str(client.name),
                euid=euid,
                space=client._experiment.space.configuration,
                version=client._experiment.version,
                mode="x",
                max_trials=client._experiment.max_trials,
                working_dir=client._experiment.working_dir,
                metadata=client._experiment.metadata,
            )
        )

    def suggest(self, request: RequestContext) -> Dict:

        # NOTE: need to fix lost trials here
        # so reschedule them
        client = build_experiment_client(request)

        if client.is_broken:
            raise BrokenExperiment()

        trial = client.suggest(**request.data)

        small_trial = self._make_trial(trial)
        return success(dict(trials=[small_trial]))

    def _make_trial(self, trial):
        # We only send the minimal amount of information to the client
        # to force the client the communicate with us
        # to avoid having invisible issues.
        data = trial.to_dict()

        small_trial = dict()
        small_trial["db_id"] = str(data["_id"])
        small_trial["params_id"] = str(data["id"])
        small_trial["params"] = data["params"]
        small_trial["working_dir"] = str(trial.get_working_dir())

        return small_trial

    def observe(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)
        storage = client.storage

        trial_id = request.data.get("trial_id")
        trial_hash = request.data.pop("trial_hash")
        experiment_name = request.data.pop("experiment_name")
        euid = request.data.pop("euid")
        results = request.data.get("results")

        trial = storage.get_trial(uid=trial_hash, experiment_uid=client._experiment.id)

        assert trial is not None, "Trial not found"
        client.observe(trial, results)

        return success(dict())

    def is_done(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        if client.is_broken:
            raise BrokenExperiment()

        return success(dict(is_done=client.is_done))

    def heartbeat(self, request: RequestContext) -> Dict:
        storage = get_storage_for_user(request)
        trial_id = request.data.get("trial_id")

        results = storage._db._db["trials"].update_one(
            {
                "_id": ObjectId(trial_id),
                "status": "reserved",
                "owner_id": request.username,
            },
            {
                "$set": {
                    "heartbeat": datetime.datetime.utcnow(),
                }
            },
        )
        return success(dict(updated=results.modified_count > 0))

    def insert(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        params = request.data.pop("params")
        results = request.data.pop("results")
        reserve = request.data.pop("reserve")

        results = client.insert(params=params, results=results, reserve=reserve)

        return success(dict(result=results))

    def fetch_noncompleted_trials(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        with_evc_tree = request.data.pop("with_evc_tree")

        results = client.fetch_noncompleted_trials(with_evc_tree=with_evc_tree)

        results = [self._make_trial(t) for t in results]
        return success(dict(result=results))

    def fetch_pending_trials(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        with_evc_tree = request.data.pop("with_evc_tree")

        results = client.fetch_pending_trials(with_evc_tree=with_evc_tree)
        results = [self._make_trial(t) for t in results]
        return success(dict(result=results))

    def fetch_trials_by_status(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        status = request.data.pop("status")
        with_evc_tree = request.data.pop("with_evc_tree")

        results = client.fetch_trials_by_status(
            status=status, with_evc_tree=with_evc_tree
        )
        results = [self._make_trial(t) for t in results]
        return success(dict(result=results))

    def get_trial(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        trial = request.data.pop("trial")
        uid = request.data.pop("uid")

        trial = client.get_trial(trial=trial, uid=uid)
        results = self._make_trial(trial)
        return success(dict(result=results))

    def fetch_trials(self, request: RequestContext) -> Dict:
        client = build_experiment_client(request)

        with_evc_tree = request.data.pop("with_evc_tree")

        results = client.fetch_trials(with_evc_tree=with_evc_tree)
        results = [self._make_trial(t) for t in results]
        return success(dict(result=results))
