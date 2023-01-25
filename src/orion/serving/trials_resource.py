"""
Module responsible for the trials/ REST endpoint
================================================
"""
import json

from falcon import Request, Response

from orion.serving.parameters import (
    retrieve_experiment,
    retrieve_trial,
    verify_query_parameters,
    verify_status,
)
from orion.serving.responses import build_trial_response, build_trials_response

SUPPORTED_PARAMETERS = ["ancestors", "status", "version"]


class TrialsResource:
    """Serves all the requests made to trials/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get_trials_in_experiment(
        self, req: Request, resp: Response, experiment_name: str
    ):
        """
        Handle GET requests for trials/:experiment where ``experiment`` is
        the user-defined name of the experiment.
        """
        verify_query_parameters(req.params, SUPPORTED_PARAMETERS)

        status = req.get_param("status", default=None)
        verify_status(status)
        version = req.get_param_as_int("version")
        with_ancestors = req.get_param_as_bool("ancestors", default=False)

        experiment = retrieve_experiment(self.storage, experiment_name, version)
        if status:
            trials = experiment.fetch_trials_by_status(status, with_ancestors)
        else:
            trials = experiment.fetch_trials(with_ancestors)

        response = build_trials_response(trials)
        resp.text = json.dumps(response)

    def on_get_trial_in_experiment(
        self, req: Request, resp: Response, experiment_name: str, trial_id: str
    ):
        """
        Handle GET requests for trials/:experiment/:trial_id where ``experiment`` is
        the user-defined name of the experiment and ``trial_id`` the id of the trial.
        """
        experiment = retrieve_experiment(self.storage, experiment_name)
        trial = retrieve_trial(experiment, trial_id)

        response = build_trial_response(trial)
        resp.text = json.dumps(response)
