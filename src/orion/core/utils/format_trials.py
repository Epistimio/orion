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
    assert len(trial.params) == len(space)
    for order, param in enumerate(trial.params):
        assert space[order].name == param.name
    return tuple([param.value for param in trial.params])


def tuple_to_trial(data, space):
    """Create a `orion.core.worker.trial.Trial` object from `data`,
    filling only parameter information from `data`.

    :param data: A tuple representing a sample point from `space`.
    :param space: Definition of problem's domain.
    :type space: `orion.algo.space.Space`
    """
    assert len(data) == len(space)
    params = [dict(
        name=space[order].name,
        type=space[order].type,
        value=data[order]
        ) for order in range(len(space))]
    return Trial(params=params)


def get_trial_results(trial):
    """Format results from a `Trial` using standard structures."""
    results = dict()
    obj = trial.objective
    results['objective'] = obj.value if obj else None
    results['constraint'] = [result.value for result in trial.results
                             if result.type == 'constraint']
    grad = trial.gradient
    results['gradient'] = tuple(grad.value) if grad else None

    return results
