# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.mutate_functions` --
Different mutate functions
===========================================================================================

.. module:: mutate_functions
    :platform: Unix
    :synopsis: Implement evolution to exploit configurations with fixed resource efficiently

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def default_mutate(multiply_factor, add_factor, volatility, search_space, old_value):
    """Get a default mutate function"""
    lower_bound = -np.inf
    upper_bound = np.inf
    if search_space.type == "real":
        if search_space.prior.name == "uniform" or \
           search_space.prior.name == "loguniform":
            lower_bound = search_space.prior.a
            upper_bound = search_space.prior.b

        factors = (1.0 / multiply_factor + (multiply_factor - 1.0 / multiply_factor) *
                   np.random.random())
        if lower_bound <= old_value * factors <= upper_bound:
            new_value = old_value * factors
        elif lower_bound > old_value * factors:
            new_value = lower_bound + volatility * np.random.random()
        else:
            new_value = upper_bound - volatility * np.random.random()
    elif search_space.type == "integer":
        if search_space.prior.name == "uniform" or \
           search_space.prior.name == "loguniform":
            lower_bound = search_space.prior.a
            upper_bound = search_space.prior.b

        factors = int(add_factor * (2 * np.random.randint(2) - 1))
        if lower_bound <= old_value + factors <= upper_bound:
            new_value = old_value + factors
        elif lower_bound > old_value + factors:
            new_value = int(lower_bound)
        else:
            new_value = int(upper_bound)
    elif search_space.type == "categorical":
        sample_index = \
            np.where(np.random.multinomial(1,
                                           list(search_space.get_prior)) == 1)[0][0]
        new_value = search_space.categories[sample_index]
    else:
        new_value = old_value

    return new_value
