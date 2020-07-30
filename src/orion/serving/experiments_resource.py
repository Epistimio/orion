# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.experiments_resource` -- Module responsible for the experiments/ REST endpoint
==================================================================================================

.. module:: experiments_resource
   :platform: Unix
   :synopsis: Serves all the requests made to experiments/ REST endpoint
"""
import json

from falcon import Request, Response
import falcon.status_codes

from orion.core.io import experiment_builder
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.worker.experiment import Experiment
from orion.storage.base import get_storage


class ExperimentsResource(object):
    """Handle requests for the experiments/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get(self, req: Request, resp: Response):
        """Handle the GET requests for experiments/"""
        experiments = self.storage.fetch_experiments({})
        leaf_experiments = _find_latest_versions(experiments)

        result = []
        for name, version in leaf_experiments.items():
            result.append({
                'name': name,
                'version': version
            })

        resp.body = json.dumps(result, indent=4)

    def on_get_experiment(self, req: Request, resp: Response, name: str):
        """
        Handle GET requests for experiments/:name where `name` is
        the user-defined name of the experiment
        """
        try:
            _verify_experiment_parameters(req.params)

            version = int(req.params['version']) if 'version' in req.params else None
            experiment = experiment_builder.build_view(name, version)

            response = {
                "name": experiment.name,
                "version": experiment.version,
                "status": _retrieve_status(experiment),
                "trialsCompleted": experiment.stats['trials_completed'] if experiment.stats else 0,
                "startTime": str(experiment.stats['start_time']) if experiment.stats else None,
                "endTime": str(experiment.stats['finish_time']) if experiment.stats else None,
                "user": experiment.metadata['user'],
                "orionVersion": experiment.metadata['orion_version'],
                "config": {
                    "maxTrials": experiment.max_trials,
                    "poolSize": experiment.pool_size,
                    "algorithm": _retrieve_algorithm(experiment),
                    "space": experiment.configuration['space']
                },
                "bestTrial": _retrieve_best_trial(experiment)
            }

            resp.body = json.dumps(response, indent=4)

        except NoConfigurationError:
            response = {
                "message": f"Experiment '{name}' does not exist"
            }
            resp.body = json.dumps(response, indent=4)
            resp.status = falcon.HTTP_404

        except ValueError as exception:
            response = {
                "message": str(exception)
            }

            resp.body = json.dumps(response)
            resp.status = falcon.HTTP_400


def _find_latest_versions(experiments):
    """Find the latest versions of the experiments"""
    leaf_experiments = {}
    for experiment in experiments:
        name = experiment['name']
        version = experiment['version']

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

    result = {'name': algorithm_name}
    result.update(experiment.algorithms.configuration[algorithm_name])
    return result


def _retrieve_best_trial(experiment: Experiment) -> dict:
    """Constructs the view of the best trial if there is one"""
    if not experiment.stats:
        return {}

    trial = experiment.get_trial(uid=experiment.stats['best_trials_id'])

    result = {'id': trial.id,
              'submitTime': str(trial.submit_time),
              'startTime': str(trial.start_time),
              'endTime': str(trial.end_time),
              'parameters': trial.params,
              'objective': trial.objective.value,
              'statistics': {statistic.name: statistic.value for statistic in trial.statistics}
              }

    return result


def _verify_experiment_parameters(params: dict):
    """
    Verifies that the parameters from the query string are supported.

    Parameters
    ----------
    params:
        A dictionary of (parameter:value)

    Raises
    ------
    ValueError
        When an unsupported parameter is found
    """
    supported_parameter = 'version'
    error_message = "Parameter '{}' is not supported. Expected parameter '{}'."

    for key in params:
        if key != supported_parameter:
            raise ValueError(error_message.format(key, supported_parameter))
