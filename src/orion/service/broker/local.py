import datetime
import logging
from typing import Dict

from bson import ObjectId

from orion.core.utils.exceptions import BrokenExperiment
from orion.service.broker.broker import (
    ExperimentBroker,
    RequestContext,
    build_experiment,
    build_experiment_client,
    get_storage_for_user,
    success,
)

log = logging.getLogger(__file__)


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
                space=client._experiment.space,
                version=client._experiment.version,
                mode="r",
                working_dir=client._experiment.working_dir,
                metadata=client._experiment.metadata,
            )
        )

    def suggest(self, request: RequestContext) -> Dict:

        # NOTE: need to fix lost trials here
        # so reschedule them

        storage = get_storage_for_user(request)
        experiment_name = request.data.pop("experiment_name")

        log.debug("Suggest %s", experiment_name)
        client = build_experiment(name=experiment_name, storage=storage)
        client.remote_mode = True

        if client.is_broken:
            raise BrokenExperiment()

        trial = client.suggest(**request.data).to_dict()

        # We only send the minimal amount of information to the client
        # to force the client the communicate with us
        # to avoid having invisible issues.
        small_trial = dict()
        small_trial["db_id"] = str(trial["_id"])
        small_trial["params_id"] = str(trial["id"])
        small_trial["params"] = str(trial["params"])

        return success(dict(trials=[small_trial]))

    def observe(self, request: RequestContext) -> Dict:
        storage = get_storage_for_user(request)

        trial_id = request.data.get("trial_id")
        euid = request.data.get("euid")
        results = request.data.get("results")

        # we do not need all the clutter from client.observe
        # 1. it makes the producer oberserve the result
        #    but the producer gets deleted after the observe
        # 2. it makes us create a trial object, fetch the previous results
        #    and then update the trial object
        #    before we can push the results to the storage
        # Push the result to the storage
        result = storage._db._db["trials"].update_one(
            {
                "_id": ObjectId(trial_id),
                "experiment": ObjectId(euid),
                "owner_id": request.username,
                "status": "reserved",  # This protects us against race conditions
            },  # Trials should be reserved when being worked on
            {
                "$push": {
                    "results": {
                        "$each": results,
                    }
                },
                "$set": {
                    "status": "completed",
                    "end_time": datetime.datetime.utcnow(),
                },
            },
        )

        if result.modified_count <= 0:
            raise ObserverError("Trial update failed, was it reserved ?")

        return success(dict())

    def is_done(self, request: RequestContext) -> Dict:
        storage = get_storage_for_user(request)
        experiment_name = request.data.pop("experiment_name")

        client = build_experiment(name=experiment_name, storage=storage)
        client.remote_mode = True

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
