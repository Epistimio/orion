"""
Module responsible for the experiments/ REST endpoint
=====================================================

Serves all the requests made to experiments/ REST endpoint.

"""
import json
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Optional

from falcon import Request, Response

from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.serving.parameters import retrieve_experiment, verify_query_parameters
from orion.serving.responses import (
    build_experiment_response,
    build_experiments_response,
)


def _compute_trial_duration(trial: Trial):
    if trial.end_time:
        return trial.end_time - trial.start_time
    elif trial.heartbeat:
        return trial.heartbeat - trial.start_time
    else:
        return timedelta()


class ExperimentStatus:
    __slots__ = (
        "trial_status_percentage",
        "eta",
        "current_execution_time",
        "whole_clock_time",
        "progress",
    )

    def __init__(self):
        self.trial_status_percentage = {status: 0.0 for status in Trial.allowed_stati}
        self.eta = timedelta()
        self.current_execution_time = timedelta()
        self.whole_clock_time = timedelta()
        self.progress = 0.0

    def to_dict(self):
        return {
            "trial_status_percentage": self.trial_status_percentage,
            "eta": str(self.eta),
            "current_execution_time": str(self.current_execution_time),
            "whole_clock_time": str(self.whole_clock_time),
            "progress": self.progress,
        }


def retrieve_experiment_status(experiment):
    trials: List[Trial] = experiment.fetch_trials(with_evc_tree=False)

    # List completed trials
    completed_trials = [trial for trial in trials if trial.status == "completed"]
    # List running trials (status new or reserved)
    running_trials = [trial for trial in trials if trial.status in ("new", "reserved")]
    # Compute average duration for completed trials
    average_completed_duration = sum(
        (_compute_trial_duration(trial) for trial in completed_trials),
        start=timedelta(),
    ) / len(completed_trials)

    exp_stat = ExperimentStatus()
    # Compute number of trials per trial status
    for status, count in Counter(trial.status for trial in trials).items():
        exp_stat.trial_status_percentage[status] = count / len(trials)
    # Compute estimated remaining time for experiment to finish
    exp_stat.eta = (
        float("+inf")
        if average_completed_duration == 0
        else average_completed_duration * len(running_trials)
    )
    # Compute current execution time
    min_exc_time = min(trial.start_time for trial in trials if trial.start_time)
    max_exc_time = datetime.fromtimestamp(0)
    for trial in trials:
        if trial.end_time:
            max_exc_time = max(max_exc_time, trial.end_time)
        elif trial.heartbeat:
            max_exc_time = max(max_exc_time, trial.heartbeat)
    exp_stat.current_execution_time = (
        timedelta() if min_exc_time > max_exc_time else max_exc_time - min_exc_time
    )
    # Compute whole clock time
    exp_stat.whole_clock_time = sum(
        (_compute_trial_duration(trial) for trial in trials), start=timedelta()
    )
    # Compute experiment progress
    exp_stat.progress = (len(trials) - len(running_trials)) / len(trials)
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
