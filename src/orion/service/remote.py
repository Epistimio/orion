from dataclasses import dataclass
import multiprocessing as mp
import logging
import traceback
from typing import Dict

from orion.client import build_experiment
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


class RemoteExperimentBroker:
    def __init__(self) -> None:
        self.manager = mp.Manager()
        self.experiments: Dict[str, ExperimentContext] = dict()
        self.service_ctx = ServiceContext()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        log.debug("Closing queues")
        for ctx in self.experiments.values():
            ctx.request_queue.put(dict(function='stop'))

        log.debug("Closing workers")
        for ctx in self.experiments.values():
            ctx.process.join()

        self.manager.shutdown()

    def new_experiment(self, token, experiment_ctx):
        log.debug("Spawning new experiment")

        experiment_ctx.request_queue = self.manager.Queue()
        experiment_ctx.result_queue = self.manager.Queue()

        experiment_ctx.process = mp.Process(
            target=experiment_worker,
            args=(self.service_ctx, experiment_ctx)
        )
        experiment_ctx.process.start()
        result = experiment_ctx.result_queue.get()

        if result['status'] == 0:
            exp_id = result['result']['experiment_id']
            self.experiments[(token, exp_id)] = experiment_ctx

        return result

    def resume_from_experiment(experiment_id):
        pass

    def resume_from_trial(trial_id):
        pass

    def experiment_request(self, token, experiment_id, function, *args, **kwargs):
        ctx = self.experiments.get((token, experiment_id))
        if ctx is None:
            ctx = self.resume_experiment()

        ctx.request_queue.put(dict(
            function=function,
            args=args,
            kwargs=kwargs
        ))

        return ctx.result_queue.get()


def experiment_worker(service_ctx: ServiceContext, exp_ctx: ExperimentContext):
    running = False
    client = None

    def success(values):
        exp_ctx.result_queue.put(dict(status=0, result=values))

    def error(exception):
        exp_ctx.result_queue.put(dict(status=1, error=str(exception)))

    try:
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

        log.debug("Building experiment")
        name = exp_ctx.config.pop('name')

        client = build_experiment(name, **exp_ctx.config, storage_instance=storage)
        success(dict(experiment_id=str(client.id)))
        running = True

    except Exception as err:
        traceback.print_exc()
        error(err)

    if client is None:
        return

    while running:
        # wait until we receive a request
        rpc_request = exp_ctx.request_queue.get(True)

        function = rpc_request.pop('function')

        if function == 'stop':
            log.debug("Stopping worker")
            break

        try:
            args = rpc_request.pop('arg')
            kwargs = rpc_request.pop('kwargs')

            result = getattr(client, function)(*args, **kwargs)
            success(result)

        except Exception as err:
            traceback.print_exc()
            error(err)

