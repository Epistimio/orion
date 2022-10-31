from __future__ import annotations

import logging
from dataclasses import dataclass

import falcon

from orion.client import ExperimentClient, build_experiment
from orion.core.io.database.mongodb import MongoDB
from orion.storage.legacy import Legacy

log = logging.getLogger(__name__)


def success(values) -> dict:
    """Create a success response"""
    return dict(status=0, result=values)


def error(exception) -> dict:
    """Create an error response"""
    return dict(status=1, error=str(exception))


@dataclass
class Database:
    database: str = "orion"
    host: str = "localhost"
    port: int = 8123


@dataclass
class Authentication:
    """Authentication configuration"""

    database: str = "users"
    host: str = "localhost"
    port: int = 8124
    usernane: str = "god"
    password: str = "god123"


@dataclass
class ServiceContext:
    """Global configuration for the service"""

    database: Database = Database()
    authentication: Authentication = Authentication()
    broker: ExperimentBroker = None
    auth = AuthenticationService = None


@dataclass
class RequestContext:
    """Request specific information useful throughout the request handling"""

    service: ServiceContext  # Service config
    username: str  # Populated on authentication
    password: str
    data: dict  # Parameter data from the request
    token: str  # Original token
    request: falcon.Request = None  # Original Request
    response: falcon.Response = None  # Response to be sent back


def get_storage_for_user(request: RequestContext):
    """Create a mongodb connection for a given user"""
    log.debug("Connecting to database")

    assert request.username is not None

    log.debug(
        "Mongo %s, %s, %s",
        request.service.database.host,
        request.service.database.port,
        request.service.database.database,
    )

    db = MongoDB(
        name=request.service.database.database,
        host=request.service.database.host,
        port=request.service.database.port,
        username=request.username,
        password=request.password,
        owner=request.username,
    )

    storage = Legacy(
        database_instance=db,
        # Skip setup, this is a shared database
        # we might not have the permissions to do the setup
        # and the setup should already be done anyway
        setup=False,
    )

    return storage


def create_experiment_client(request: RequestContext) -> ExperimentClient:
    """Create an experiment client by creating the initial experiment"""

    storage = get_storage_for_user(request)

    log.debug("Building experiment %s", request.data)

    # ignore the storage config
    _ = request.data.pop("storage", None)

    client = build_experiment(
        **request.data,
        storage=storage,
        # if we keep the username on the storage that would prevent
        # more pervasive modification
        # username=request.username,
    )
    client.remote_mode = True
    return client


def retrieve_experiment_client(request: RequestContext, name) -> ExperimentClient:
    """Create an experiment client by retrieving an already created experiment"""

    storage = get_storage_for_user(request)

    log.debug("Building experiment")

    # ignore the storage config
    _ = request.data.pop("storage", None)

    client = build_experiment(
        name=name,
        storage=storage,
        # if we keep the username on the storage that would prevent
        # more pervasive modification
        # username=request.username,
    )
    client.remote_mode = True
    return client


class ExperimentBroker:
    """Broker allocates the necessary resources to run the HPO on our side.
    trials are done on the user' side

    """

    def __init__(self, ctx: ServiceContext) -> None:
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        pass

    def new_experiment(self, request: RequestContext):
        """Create a new experiment"""

    def suggest(self, request: RequestContext):
        """Suggest and reserve a new trial

        See `~orion.client.ExperimentClient.suggest`
        """

    def observe(self, request: RequestContext):
        """observe and release a trial

        See `~orion.client.ExperimentClient.observe`
        """

    def is_done(self, request: RequestContext):
        """return true if the experiment is done

        See `~orion.client.ExperimentClient.is_done`
        """

    def heartbeat(self, request: RequestContext):
        """update the heartbeat of a trial

        See `~orion.client.ExperimentClient.heartbeat`
        """

    def release(self, request: RequestContext):
        """release a trial

        See `~orion.client.ExperimentClient.release`
        """

    def insert(self, request: RequestContext):
        """insert a trial

        See `~orion.client.ExperimentClient.insert`
        """

    def fetch_noncompleted_trials(self, request: RequestContext):
        """Fetch non completed trials

        See `~orion.client.ExperimentClient.fetch_noncompleted_trials`
        """

    def fetch_pending_trials(self, request: RequestContext):
        """Fetch pending trials

        See `~orion.client.ExperimentClient.fetch_pending_trials`
        """

    def fetch_trials_by_status(self, request: RequestContext):
        """Fetch trials by status

        See `~orion.client.ExperimentClient.fetch_trials_by_status`
        """

    def get_trial(self, request: RequestContext):
        """Retrieve a trial

        See `~orion.client.ExperimentClient.get_trial`
        """

    def fetch_trials(self, request: RequestContext):
        """Fetch trials

        See `~orion.client.ExperimentClient.fetch_trials`
        """
