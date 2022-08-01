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

        client = build_experiment(name=experiment_name, storage_instance=storage)
        client.remote_mode = True

        trial = client.suggest(**request.data).to_dict()

        trial["experiment"] = str(trial["experiment"])
        trial.pop("heartbeat")
        trial.pop("submit_time")
        trial.pop("start_time")
        print(trial)

        return success(dict(trials=[trial]))
