# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.format_trials` -- Utility functions for formatting data
==============================================================================

.. module:: format_trials
   :platform: Unix
   :synopsis: Conversion functions between various data types used in
      framework's ecosystem.

"""

from orion.core.worker.trial import Trial


def trial_to_tuple(trial, space):
    """Extract a parameter tuple from a `orion.core.worker.trial.Trial`.

    The order within the tuple is dictated by the defined
    `orion.algo.space.Space` object.
    """
    params = {param.name: param.value for param in trial.params}
    trial_keys = set(params.keys())
    space_keys = set(space.keys())
    if trial_keys != space_keys:
        raise ValueError(""""
The trial {} has wrong params:
Trial params: {}
Space dims: {}""".format(trial.id, sorted(trial_keys), sorted(space_keys)))

    return tuple(params[name] for name in space.keys())


def tuple_to_trial(data, space):
    """Create a `orion.core.worker.trial.Trial` object from `data`,
    filling only parameter information from `data`.

    :param data: A tuple representing a sample point from `space`.
    :param space: Definition of problem's domain.
    :type space: `orion.algo.space.Space`
    """
    assert len(data) == len(space)
    params = []
    for i, dim in enumerate(space.values()):
        params.append(dict(
            name=dim.name,
            type=dim.type,
            value=data[i]
            ))
    return Trial(params=params)


def get_trial_results(trial):
    """Format results from a `Trial` using standard structures."""
    results = dict()

    lie = trial.lie
    objective = trial.objective

    if lie:
        results['objective'] = lie.value
    elif objective:
        results['objective'] = objective.value
    else:
        results['objective'] = None

    results['constraint'] = [result.value for result in trial.results
                             if result.type == 'constraint']
    grad = trial.gradient
    results['gradient'] = tuple(grad.value) if grad else None

    return results


def standard_param_name(name):
    """Convert parameter name to namespace format"""
    return name.lstrip("/").lstrip("-").replace("-", "_")
