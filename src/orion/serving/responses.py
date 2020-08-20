# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.responses` -- Helpers for building responses according to the specification
===============================================================================================

.. module:: responses
   :platform: Unix
   :synopsis: Offers functions and attributes to generate response objects according
      to the API specification
"""
from orion.core.worker.trial import Trial

ERROR_EXPERIMENT_NOT_FOUND = 'Experiment not found'
ERROR_INVALID_PARAMETER = 'Invalid parameter'
ERROR_TRIAL_NOT_FOUND = 'Trial not found'


def build_trial_response(trial: Trial) -> dict:
    """
    Build the response representing a trial according to the API specification.

    Parameters
    ----------
    trial: Trial
        The trial to return for the API.

    Returns
    -------
    A JSON-serializable dict representing the given trial.

    """
    return {'id': trial.id,
            'submitTime': str(trial.submit_time),
            'startTime': str(trial.start_time),
            'endTime': str(trial.end_time),
            'parameters': trial.params,
            'objective': trial.objective.value,
            'statistics': {statistic.name: statistic.value for statistic in trial.statistics}}
