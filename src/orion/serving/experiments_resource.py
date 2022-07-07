"""
Module responsible for the experiments/ REST endpoint
=====================================================

Serves all the requests made to experiments/ REST endpoint.

"""
import json
from typing import Optional

from falcon import Request, Response

from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.serving.parameters import retrieve_experiment, verify_query_parameters
from orion.serving.responses import (
    build_experiment_response,
    build_experiments_response,
)


class ExperimentsResource:
    """Handle requests for the experiments/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get(self, req: Request, resp: Response):
        """Handle the GET requests for experiments/"""
        experiments = self.storage.fetch_experiments({})
        leaf_experiments = _find_latest_versions(experiments)

        response = build_experiments_response(leaf_experiments)
        resp.body = json.dumps(response)

    def on_get_experiment(self, req: Request, resp: Response, name: str):
        """
        Handle GET requests for experiments/:name where `name` is
        the user-defined name of the experiment
        """
        verify_query_parameters(req.params, ["version"])
        version = req.get_param_as_int("version")
        experiment = retrieve_experiment(self.storage, name, version)

        status = _retrieve_status(experiment)
        algorithm = _retrieve_algorithm(experiment)
        best_trial = _retrieve_best_trial(experiment)

        response = build_experiment_response(experiment, status, algorithm, best_trial)
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
