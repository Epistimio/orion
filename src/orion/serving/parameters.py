"""
Common code for verifying query parameters
==========================================
"""
from typing import List, Optional

import falcon

from orion.benchmark import Benchmark
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.core.io import experiment_builder
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.serving.responses import (
    ERROR_BENCHMARK_NOT_FOUND,
    ERROR_BENCHMARK_STUDY_NOT_FOUND,
    ERROR_EXPERIMENT_NOT_FOUND,
    ERROR_INVALID_PARAMETER,
    ERROR_TRIAL_NOT_FOUND,
)


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
            raise falcon.HTTPBadRequest(
                title=ERROR_INVALID_PARAMETER, description=description
            )


def verify_status(status):
    """Verifies that the given trial status is supported. Raises falcon.HTTPBadRequest otherwise"""
    if status and status not in Trial.allowed_stati:
        description = 'The "status" parameter is invalid. '
        description += "The value of the parameter must be one of {}".format(
            list(Trial.allowed_stati)
        )

        raise falcon.HTTPBadRequest(
            title=ERROR_INVALID_PARAMETER, description=description
        )


def _compose_error_message(key: str, supported_parameters: list):
    """Creates the error message depending on the number of supported parameters available."""
    error_message = f'Parameter "{key}" is not supported. Expected '

    if len(supported_parameters) > 1:
        supported_parameters.sort()
        error_message += f"one of {supported_parameters}."
    else:
        error_message += f'parameter "{supported_parameters[0]}".'

    return error_message


def retrieve_experiment(
    storage, experiment_name: str, version: int = None
) -> Optional[Experiment]:
    """
    Retrieve an experiment from the database with the given name and version.

    Raises
    ------
    falcon.HTTPNotFound
        When the experiment doesn't exist
    """
    try:
        experiment = experiment_builder.load(experiment_name, version, storage=storage)
        if version and experiment.version != version:
            raise falcon.HTTPNotFound(
                title=ERROR_EXPERIMENT_NOT_FOUND,
                description=f'Experiment "{experiment_name}" has no version "{version}"',
            )
        return experiment
    except NoConfigurationError:
        raise falcon.HTTPNotFound(
            title=ERROR_EXPERIMENT_NOT_FOUND,
            description=f'Experiment "{experiment_name}" does not exist',
        )


def retrieve_trial(experiment: Experiment, trial_id: str):
    """
    Retrieves the trial for the given id in the experiment

    Parameters
    ----------
    experiment: Experiment
        The experiment containing the trial.

    trial_id: str
        The id of the trial

    Raises
    ------
    falcon.HTTPNotFound
        When the trial doesn't exist in the experiment.

    """
    trial = experiment.get_trial(uid=trial_id)
    if not trial:
        raise falcon.HTTPNotFound(
            title=ERROR_TRIAL_NOT_FOUND,
            description=f'Trial "{trial_id}" does not exist',
        )
    return trial


def retrieve_benchmark(
    storage,
    benchmark_name: str,
    assessment: Optional[str] = None,
    task: Optional[str] = None,
    algorithms: Optional[List[str]] = None,
) -> Optional[Benchmark]:
    """
    Retrieve an benchmark from the database with the given name.

    Parameters
    ----------
    benchmark_name: str
        Name of the benchmark to fetch.
    assessment: str or None, optional
        Filter analysis and only return those for the given assessment name.
        This value will only be validated, no analysis is computed here.
    task: str or None, optional
        Filter analysis and only return those for the given task name.
        This value will only be validated, no analysis is computed here.
    algorithms: list of str or None, optional
        Compute analysis only on specified algorithms. Compute on all otherwise.
        This value will only be validated, no analysis is computed here.

    Raises
    ------
    falcon.HTTPNotFound
        When the benchmark doesn't exist
    """
    try:
        benchmark = get_or_create_benchmark(storage, benchmark_name)
        benchmark.validate_assessment(assessment)
        benchmark.validate_task(task)
        benchmark.validate_algorithms(algorithms)
        return benchmark
    except ValueError as e:
        raise falcon.HTTPNotFound(
            title=ERROR_BENCHMARK_STUDY_NOT_FOUND,
            description=str(e),
        )
    except NoConfigurationError:
        raise falcon.HTTPNotFound(
            title=ERROR_BENCHMARK_NOT_FOUND,
            description=f'Benchmark "{benchmark_name}" does not exist',
        )
