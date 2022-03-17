# -*- coding: utf-8 -*-
"""
Module responsible for the benchmarks/ REST endpoint
=====================================================

Serves all the requests made to benchmarks/ REST endpoint.

"""
import time
import json
from typing import Optional

from falcon import Request, Response

from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.serving.parameters import retrieve_benchmark, verify_query_parameters
from orion.serving.responses import (
    build_benchmark_response,
    build_benchmarks_response,
)
from orion.storage.base import get_storage


class BenchmarksResource(object):
    """Handle requests for the benchmarks/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get(self, req: Request, resp: Response):
        """Handle the GET requests for benchmarks/"""
        benchmarks = self.storage.fetch_benchmark({})

        response = build_benchmarks_response(benchmarks)
        resp.body = json.dumps(response)

    def on_get_benchmark(self, req: Request, resp: Response, name: str):
        """
        Handle GET requests for benchmarks/:name where `name` is
        the user-defined name of the benchmark
        """
        verify_query_parameters(req.params, ["assessment", "task", "algorithms"])
        assessment = req.get_param("assessment")
        task = req.get_param("task")
        algorithms = req.get_param_as_list("algorithms")
        benchmark = retrieve_benchmark(
            name, assessment=assessment, task=task, algorithms=algorithms
        )

        response = build_benchmark_response(benchmark, assessment, task, algorithms)
        resp.body = json.dumps(response)


def _find_latest_versions(experiments):
    """Find the latest versions of the experiments"""
    leaf_experiments = {}
    for experiment in experiments:
        name = experiment["name"]
        version = experiment["version"]

        if name in leaf_experiments:
            leaf_experiments[name] = max(leaf_experiments[name], version)
        else:
            leaf_experiments[name] = version
    return leaf_experiments


def _retrieve_status(experiment: Experiment) -> str:
    """
    Determines the status of an experiment.

    Returns
    -------
        "done" if the experiment is complete, otherwise "not done".
    """
    return "done" if experiment.is_done else "not done"


def _retrieve_algorithm(experiment: Experiment) -> dict:
    """Populates the `algorithm` key with the configuration of the experiment's algorithm."""
    algorithm_name = list(experiment.algorithms.configuration.keys())[0]

    result = {"name": algorithm_name}
    result.update(experiment.algorithms.configuration[algorithm_name])
    return result


def _retrieve_best_trial(experiment: Experiment) -> Optional[Trial]:
    """Constructs the view of the best trial if there is one"""
    if not experiment.stats:
        return None

    return experiment.get_trial(uid=experiment.stats.best_trials_id)
