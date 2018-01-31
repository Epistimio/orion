# -*- coding: utf-8 -*-
"""
:mod:`metaopt.core.utils.format` -- Utility functions for formatting data
=========================================================================

.. module:: format
   :platform: Unix
   :synopsis: Conversion functions between various data types used in
      framework's ecosystem.

"""

from metaopt.core.worker.trial import Trial


def trial_to_tuple(trial, space):
    """Extract a parameter tuple from a `metaopt.core.worker.trial.Trial`.

    The order within the tuple is dictated by the defined
    `metaopt.algo.space.Space` object.
    """
    assert len(trial.params) == len(space)
    for order, param in enumerate(trial.params):
        assert space[order].name == param.name
    return tuple([param.value for param in trial.params])


def tuple_to_trial(data, space):
    """Create a `metaopt.core.worker.trial.Trial` object from `data`,
    filling only parameter information from `data`.

    :param data: A tuple representing a sample point from `space`.
    :param space: Definition of problem's domain.
    :type space: `metaopt.algo.space.Space`
    """
    assert len(data) == len(space)
    params = [dict(
        name=space[order].name,
        type=space[order].type,
        value=data[order]
        ) for order in range(len(space))]
    return Trial(params=params)
