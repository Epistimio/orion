"""
:mod:`orion.serving.parameters` -- Common code for verifying query parameters
=============================================================================

.. module:: parameters
   :platform: Unix
   :synopsis: Common code related to query parameters verification
"""
from typing import Optional

import falcon

from orion.core.io import experiment_builder
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial


def verify_query_parameters(parameters: dict, supported_parameters: list):
    """
    Verifies that the parameters given in the input dictionary are all supported.

    Parameter
    ---------
    parameters
        The dictionary of parameters to verify in the format ``parameter_name:value``.
    supported_parameters
        The list of parameters that are supported.

    Raises
    ------
    falcon.HTTPBadRequest
        When a parameter is not listed in the supported parameters.
    """
    for parameter in parameters:
        if parameter not in supported_parameters:
            description = _compose_error_message(parameter, supported_parameters)
            raise falcon.HTTPBadRequest('Invalid parameter', description)


def verify_status(status):
    """Verifies that the given trial status is supported. Raises falcon.HTTPBadRequest otherwise"""
    if status and status not in Trial.allowed_stati:
        description = 'The "status" parameter is invalid. '
        description += 'The value of the parameter must be one of {}'.format(
            list(Trial.allowed_stati))

        raise falcon.HTTPBadRequest('Invalid parameter', description)


def _compose_error_message(key: str, supported_parameters: list):
    """Creates the error message depending on the number of supported parameters available."""
    error_message = f'Parameter "{key}" is not supported. Expected '

    if len(supported_parameters) > 1:
        supported_parameters.sort()
        error_message += f'one of {supported_parameters}.'
    else:
        error_message += f'parameter "{supported_parameters[0]}".'

    return error_message


def retrieve_experiment(experiment_name: str, version: int = None) -> Optional[Experiment]:
    """
    Retrieve an experiment from the database with the given name and version.

    Raises
    ------
    falcon.HTTPNotFound
        When the experiment doesn't exist
    """
    try:
        experiment = experiment_builder.build_view(experiment_name, version)
        if version and experiment.version != version:
            raise falcon.HTTPNotFound(
                title='Experiment not found',
                description=f'Experiment "{experiment_name}" has no version "{version}"')
        return experiment
    except NoConfigurationError:
        raise falcon.HTTPNotFound(title='Experiment not found',
                                  description=f'Experiment "{experiment_name}" does not exist')


def retrieve_trial(experiment, trial_id):
    """
    Retrieves the trial for the given id in the experiment

    Parameters
    ----------
    experiment: Experiment
        The experiment containing the trial.

    trial_id: int
        The id of the trial

    Raises
    ------
    falcon.HTTPNotFound
        When the trial doesn't exist in the experiment.

    """
    trial = experiment.get_trial(uid=trial_id)
    if not trial:
        raise falcon.HTTPNotFound(title='Trial not found',
                                  description=f'Trial "{trial_id}" does not exist')
    return trial
