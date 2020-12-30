# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.mutate_functions` --
Different mutate functions: large-scale evolution of image classifiers
===========================================================================================

.. module:: mutate_functions
    :platform: Unix
    :synopsis: Implement evolution to exploit configurations with fixed resource efficiently

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def default_mutate(search_space, old_value, **kwargs):
    """Get a default mutate function"""
    multiply_factor = kwargs.pop("multiply_factor", 3.0)
    add_factor = kwargs.pop("add_factor", 1)
    volatility = kwargs.pop("volatility", 0.001)
    if search_space.type == "real":
        lower_bound, upper_bound = search_space.interval()
        factors = (
            1.0 / multiply_factor
            + (multiply_factor - 1.0 / multiply_factor) * np.random.random()
        )
        if lower_bound <= old_value * factors <= upper_bound:
            new_value = old_value * factors
        elif lower_bound > old_value * factors:
            new_value = lower_bound + volatility * np.random.random()
        else:
            new_value = upper_bound - volatility * np.random.random()
    elif search_space.type == "integer":
        lower_bound, upper_bound = search_space.interval()
        factors = int(add_factor * (2 * np.random.randint(2) - 1))
        if lower_bound <= old_value + factors <= upper_bound:
            new_value = int(old_value) + factors
        elif lower_bound > old_value + factors:
            new_value = int(lower_bound)
        else:
            new_value = int(upper_bound)
    elif search_space.type == "categorical":
        sample_index = np.where(
            np.random.multinomial(1, list(search_space.get_prior)) == 1
        )[0][0]
        new_value = int(search_space.categories[sample_index])
    else:
        new_value = old_value
    return new_value
