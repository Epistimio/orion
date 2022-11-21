"""
Module responsible for the experiments/ REST endpoint
=====================================================

Serves all the requests made to experiments/ REST endpoint.

"""
import json
from collections import Counter
from datetime import timedelta
from typing import List, Optional

from falcon import Request, Response

from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.serving.parameters import retrieve_experiment, verify_query_parameters
from orion.serving.responses import (
    build_experiment_response,
    build_experiments_response,
)


class ExperimentStatus:
    __slots__ = (
        "nb_trials",
        "trial_status_count",
        "eta",
        "current_execution_time",
        "whole_clock_time",
        "progress",
    )

    def __init__(self):
        self.nb_trials = 0
        self.trial_status_count = {status: 0.0 for status in Trial.allowed_stati}
        self.eta = timedelta()
        self.current_execution_time = timedelta()
        self.whole_clock_time = timedelta()
        self.progress = 0.0

    def to_dict(self):
        return {
            "nb_trials": self.nb_trials,
            "trial_status_count": self.trial_status_count,
            "eta": str(self.eta),
            "current_execution_time": str(self.current_execution_time),
            "whole_clock_time": str(self.whole_clock_time),
            "progress": self.progress,
        }


def retrieve_experiment_status(experiment):
    trials: List[Trial] = experiment.fetch_trials(with_evc_tree=False)
    st = experiment.stats

    # List completed trials
    completed_trials = [trial for trial in trials if trial.status == "completed"]
    # List running trials (status new or reserved)
    running_trials = [trial for trial in trials if trial.status in ("new", "reserved")]

    exp_stat = ExperimentStatus()
    # Register trials count
    exp_stat.nb_trials = len(trials)
    # Compute number of trials per trial status
    exp_stat.trial_status_count = Counter(trial.status for trial in trials)
    # Compute estimated remaining time for experiment to finish
    exp_stat.eta = (st.duration / len(completed_trials)) * len(running_trials)
    # Compute current execution time
    exp_stat.current_execution_time = st.duration
    # Compute whole clock time
    exp_stat.whole_clock_time = st.whole_clock_time
    # Compute experiment progress
    exp_stat.progress = len(completed_trials) / len(trials)
    return exp_stat


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

    def on_get_experiment_status(self, req: Request, resp: Response, name: str):
        """
        Handle GET requests for experiments/status/:name where `name` is
        the user-defined name of the experiment
        """
        verify_query_parameters(req.params, ["version"])
        version = req.get_param_as_int("version")
        experiment = retrieve_experiment(self.storage, name, version)
        resp.body = json.dumps(retrieve_experiment_status(experiment).to_dict())


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
