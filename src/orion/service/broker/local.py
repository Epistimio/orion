import logging
from typing import Dict

from orion.service.broker.broker import ServiceContext, ExperimentContext, build_experiment_client, ExperimentBroker


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

    """
    def __init__(self) -> None:
        self.service_ctx = ServiceContext()

    def new_experiment(self, token, experiment_ctx: ExperimentContext) -> Dict:
        log.debug("Spawning new experiment")

        client = build_experiment_client(
            self.service_ctx,
            experiment_ctx
        )

        return success(dict(experimend_id=str(client.id)))
