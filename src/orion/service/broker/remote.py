import datetime
import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Dict

from orion.service.broker.broker import (
    ExperimentBroker,
    RequestContext,
    build_experiment_client,
)

log = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Context used to fetch the process associated with a given experiment"""

    request_queue: mp.Queue = None
    result_queue: mp.Queue = None
    process: mp.Process = None
    last_active: datetime.datetime = None


class RemoteExperimentBroker(ExperimentBroker):
    """Creates an experiment client in its own process.
    All the requests from a single client are processed sequentially by a single process,
    this means there are no race conditions or locks necessary.

    Because the operations execute a lot of IO we might be able to have a lot more processes running
    than we have cores, given the fork stays pretty light and does not copy too much from the parent.

    This is harder to scale because each users need to be routed to the same server all the time.
    in the worst case the process can still be recreated though (i.e if the original server dies)
    """

    def __init__(self) -> None:
        self.manager = mp.Manager()
        self.experiments: Dict[str, ExperimentContext] = dict()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        log.debug("Closing queues")
        for ctx in self.experiments.values():
            ctx.request_queue.put(dict(function="stop"))

        log.debug("Closing workers")
        for ctx in self.experiments.values():
            ctx.process.join()

        self.manager.shutdown()

    def new_experiment(self, request: RequestContext):
        log.debug("Spawning new experiment")

        experiment = ExperimentContext()
        experiment.request_queue = self.manager.Queue()
        experiment.result_queue = self.manager.Queue()

        experiment.process = mp.Process(
            target=experiment_worker, args=(request, experiment)
        )
        experiment.process.start()
        result = experiment.result_queue.get()

        if result["status"] == 0:
            exp_id = result["result"]["experiment_name"]
            self.experiments[(request.token, exp_id)] = experiment

        return result

    def suggest(self, request: RequestContext):
        experiment_name = request.data.pop("experiment_name")
        experiment = self.experiments.get((request.token, experiment_name), None)

        if experiment is None:
            # here we should try to resume from the experiment id
            # if the experiment already exists it will work
            # if not an error will be raised
            result = self.new_experiment(request)

            if result["status"] != 0:
                raise RuntimeError(
                    f"Need to create an experiment first: {result['error']}"
                )

        experiment.request_queue.put(dict(function="suggest", kwargs=request.data))

        trial = experiment.result_queue.get()["result"].to_dict()

        trial["experiment"] = str(trial["experiment"])
        trial.pop("heartbeat")
        trial.pop("submit_time")
        trial.pop("start_time")

        return dict(status=0, result=dict(trials=[trial]))

    def resume_from_experiment(experiment_name):
        pass

    def resume_from_trial(trial_id):
        pass

    def experiment_request(self, token, experiment_name, function, *args, **kwargs):
        ctx = self.experiments.get((token, experiment_name))
        if ctx is None:
            ctx = self.resume_experiment()

        ctx.request_queue.put(dict(function=function, args=args, kwargs=kwargs))

        return ctx.result_queue.get()


def experiment_worker(request: RequestContext, experiment: ExperimentContext):
    running = False
    client = None

    def success(values):
        experiment.result_queue.put(dict(status=0, result=values))

    def error(exception):
        experiment.result_queue.put(dict(status=1, error=str(exception)))

    try:
        client = build_experiment_client(request)
        success(dict(experiment_name=str(client.name)))
        running = True

    except Exception as err:
        traceback.print_exc()
        error(err)

    if client is None:
        return

    while running:
        # wait until we receive a request
        rpc_request = experiment.request_queue.get(True)

        function = rpc_request.pop("function")

        if function == "stop":
            log.debug("Stopping worker")
            break

        try:
            args = rpc_request.pop("arg", [])
            kwargs = rpc_request.pop("kwargs", dict())

            result = getattr(client, function)(*args, **kwargs)
            success(result)

        except Exception as err:
            traceback.print_exc()
            error(err)
