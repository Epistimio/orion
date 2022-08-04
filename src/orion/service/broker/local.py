import logging
from typing import Dict

from orion.service.broker.broker import (
    ExperimentBroker,
    RequestContext,
    build_experiment,
    build_experiment_client,
    get_storage_for_user,
)

log = logging.getLogger(__file__)


def success(values) -> Dict:
    return dict(status=0, result=values)


def error(exception) -> Dict:
    return dict(status=1, error=str(exception))


class LocalExperimentBroker(ExperimentBroker):
    """Creates an experiment client for each request and process the request.
    Parallel requests will instantiate different clients which might generate race conditions,
    locks are required.

    The everything keeps being reinstantiated.

    This is easy to scale because nothing is kept between request,
    we can just load balance the request through n servers.

    """

    def new_experiment(self, request: RequestContext) -> Dict:
        log.debug("Spawning new experiment")

        client = build_experiment_client(request)

        return success(dict(experiment_name=str(client.name)))

    def suggest(self, request: RequestContext):
        storage = get_storage_for_user(request)
        experiment_name = request.data.pop("experiment_name")

        log.debug("Suggest %s", experiment_name)
        client = build_experiment(name=experiment_name, storage=storage)
        client.remote_mode = True

        trial = client.suggest(**request.data).to_dict()

        trial["experiment"] = str(trial["experiment"])

        # Fix Json encoding issues
        trial.pop("heartbeat")
        trial.pop("submit_time")
        trial.pop("start_time")
        trial["_id"] = str(trial["_id"])

        return success(dict(trials=[trial]))

    def observe(self, request: RequestContext):
        storage = get_storage_for_user(request)

        # TODO: fix this, this makes us create
        experiment_name = request.data.pop("experiment_name")
        client = build_experiment(name=experiment_name, storage=storage)

        client.remote_mode = True

        trial_id = request.data.get("trial_id")
        results = request.data.get("results")
        experiment_id = client._experiment.id

        import datetime

        # we don not need all the clutter from client.oberserve
        # 1. it makes the producer oberserve the result
        #    but the producer gets deleted after the oberserve
        # 2. it makes us create a trial object, fetch the previous results
        #    and then update the trial object
        #    before we can push the results to the storage
        # Push the result to the storage
        result = storage._db._db["trials"].update_one(
            {
                "id": trial_id,
                "experiment": experiment_id,
                "status": "reserved",
            },
            {
                "$push": {
                    "params": {
                        "$each": results,
                    }
                },
                "$set": {
                    "status": "completed",
                    "end_time": datetime.datetime.utcnow(),
                },
            },
        )

        assert result.modified_count > 0

        # print(client._experiment)
        # print(client._experiment.id)

        # trial = Trial(id_override=trial_id, experiment=client._experiment.id)
        # log.debug("Observe %s %s %s", trial_id, trial._id, trial.experiment)

        # client.observe(trial, results).to_dict()
        return success(dict())
