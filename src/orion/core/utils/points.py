# -*- coding: utf-8 -*-
"""
Utility functions for manipulating trial points
===============================================

Conversion functions between higher shape points and lists.

"""


def regroup_dims(point, space):
    """Take a list of items representing a point and regroup them appropriately as
    a point from `space`.

    Parameters
    ----------
    point: array
        Points to be regrouped.
    space: `orion.algo.space.Space`
        The optimization space.

    Returns
    -------
    list or tuple

    """
    regrouped = []
    idx = 0

    for dimension in space.values():
        shape = dimension.shape
        if shape:
            assert len(shape) == 1
            next_dim = idx + shape[0]
            regrouped.append(tuple(point[idx:next_dim]))
            idx = next_dim
        else:
            regrouped.append(point[idx])
            idx += 1

    if regrouped not in space:
        raise AttributeError(
            "The point {} is not a valid point of space {}".format(point, space)
        )

    return regrouped


def flatten_dims(point, space):
    """Flatten `point` in `space` and convert it to a list.

    Parameters
    ----------
    point: array
        Points to be regrouped.
    space: `orion.algo.space.Space`
        The optimization space.

    Returns
    -------
    list

    """
    flattened = []

    for subpoint, dimension in zip(point, space.values()):
        shape = dimension.shape
        if shape:
            assert len(shape) == 1
            flattened.extend(subpoint)
        else:
            flattened.append(subpoint)

    return flattened
