from dataclasses import dataclass
import multiprocessing as mp
import datetime
import logging

from orion.client import build_experiment, ExperimentClient
from orion.storage.legacy import Legacy
from orion.core.io.database.mongodb import MongoDB

log = logging.getLogger(__file__)

class ServiceContext:
    database: str = 'orion'
    host: str = '192.168.0.116'
    port: int = 8124


@dataclass
class ExperimentContext:
    username: str
    password: str
    config: dict
    request_queue: mp.Queue = None
    result_queue: mp.Queue = None
    process: mp.Process = None
    last_active: datetime.datetime = None


def get_storage_for_user(service_ctx: ServiceContext, exp_ctx: ExperimentContext):
    log.debug("Connecting to database")
    db = MongoDB(
        name=service_ctx.database,
        host=service_ctx.host,
        port=service_ctx.port,
        username=exp_ctx.username,
        password=exp_ctx.password
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


def build_experiment_client(service_ctx: ServiceContext, exp_ctx: ExperimentContext) -> ExperimentClient:
    """Build an experiment client in a multiuser setting (i.e without relying on singletons)"""

    storage = get_storage_for_user(service_ctx, exp_ctx)

    log.debug("Building experiment")
    return build_experiment(**exp_ctx.config, storage_instance=storage)


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

    def new_experiment(self, token, experiment_ctx):
        pass


