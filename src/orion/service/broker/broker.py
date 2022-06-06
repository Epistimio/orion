from dataclasses import dataclass
import multiprocessing as mp
import logging

import falcon

from orion.client import build_experiment, ExperimentClient
from orion.storage.legacy import Legacy
from orion.core.io.database.mongodb import MongoDB

log = logging.getLogger(__file__)


@dataclass
class ServiceContext:
    """Global configuration for the service"""
    database: str = 'orion'
    host: str = '192.168.0.116'
    port: int = 8124


@dataclass
class RequestContext:
    """Request specific information useful throughout the request handling"""
    service: ServiceContext             # Service config
    username: str                       # Populated on authentication
    password: str
    data: dict                          # Parameter data from the request
    token: str                          # Original token
    request: falcon.Request = None      # Original Request
    response: falcon.Response = None    # Reponse to be sent back


def trial_to_json(t):
    pass


def get_storage_for_user(request: RequestContext):
    log.debug("Connecting to database")
    db = MongoDB(
        name=request.service.database,
        host=request.service.host,
        port=request.service.port,
        username=request.username,
        password=request.password
    )

    # this bypass the Storage singleton logic
    log.debug("Initializing storage")
    storage = Legacy(
        database_instance=db,
        # Skip setup, this is a shared database
        # we might not have the permissions to do the setup
        # and the setup should already be done anyway
        setup=False
    )

    return storage


def build_experiment_client(request: RequestContext) -> ExperimentClient:
    """Build an experiment client in a multiuser setting (i.e without relying on singletons)"""

    storage = get_storage_for_user(request)

    log.debug("Building experiment")
    client = build_experiment(**request.data, storage_instance=storage, username=request.username)
    client.remote_mode = True
    return client


class ExperimentBroker:
    """Broker allocates the necessary resources to run the HPO on our side.
    trials are done on the user' side

    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        pass

    def new_experiment(self, experiment_ctx):
        pass

    def suggest(self, experiment_ctx):
        pass
