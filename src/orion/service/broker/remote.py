import multiprocessing as mp
import logging
import traceback
from typing import Dict

from orion.service.broker.broker import ServiceContext, ExperimentContext, build_experiment_client, ExperimentBroker


log = logging.getLogger(__file__)



class RemoteExperimentBroker(ExperimentBroker):
    """Creates an experiment client in its own process.
    All the requests from a single client are processed sequentially by a single process,
    this means there are no race conditions or locks necessary.

    Because the operations execute a lot of IO we might be able to have a lot more processes running
    than we have cores, given the fork stays pretty light and does not copy too much from the parent.
    """
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
        client = build_experiment_client(service_ctx, exp_ctx)
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

