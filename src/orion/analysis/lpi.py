# -*- coding: utf-8 -*-
"""
:mod:`orion.analysis.lpi` -- Provide tools to calculate Local Parameter Importance
==================================================================================

.. module:: orion.analysis.lpi
   :platform: Unix
   :synopsis: Provide tools to calculate Local Parameter Importance
"""
import numpy
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from orion.core.worker.transformer import build_required_space

_regressors_ = {
    "AdaBoostRegressor": AdaBoostRegressor,
    "BaggingRegressor": BaggingRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}


def train_regressor(regressor_name, data, **kwargs):
    """Train regressor model

    Parameters
    ----------
    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)
    trials: DataFrame or dict
        A dataframe of trials containing, at least, the columns 'objective' and 'id'. Or a dict
        equivalent.

    **kwargs
        Arguments for the regressor model.

    """
    if regressor_name not in _regressors_:
        raise ValueError(
            f"{regressor_name} is not a supported regressor. "
            f"Did you mean any of theses: list(_regressors_.keys())"
        )

    regressor = _regressors_[regressor_name](**kwargs)
    return regressor.fit(data[:, :-1], data[:, -1])


def to_numpy(trials, space):
    """Convert trials in DataFrame to Numpy array of (params + objective)"""
    return trials[list(space.keys()) + ["objective"]].to_numpy()


def flatten(trials_array, flattened_space):
    """Flatten dimensions"""
    flattened_points = numpy.array(
        [flattened_space.transform(point[:-1]) for point in trials_array]
    )

    return numpy.concatenate((flattened_points, trials_array[:, -1:]), axis=1)


def make_grid(point, space, model, n):
    """Build a grid based on point.

    The shape of the grid will be
        (number of hyperparameters,
         number of points ``n``,
         number of hyperparameters + 1)

    Last column is the objective predicted by the model for a given point.

    Parameters
    ----------
    point: numpy.ndarray
        A tuple representation of the best trials, (hyperparameters + objective)
    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)
    trials: DataFrame or dict
        A dataframe of trials containing, at least, the columns 'objective' and 'id'. Or a dict
        equivalent.

    **kwargs
        Arguments for the regressor model.

    """
    grid = numpy.zeros((len(space), n, len(space) + 1))
    for i, dim in enumerate(space.values()):
        grid[i, :, :] = point
        grid[i, :, i] = numpy.linspace(*dim.interval(), num=n)
        grid[i, :, -1] = model.predict(grid[i, :, :-1])
    return grid


def compute_variances(grid):
    """Compute variance for each hyperparameters"""
    return grid[:, :, -1].var(axis=1)


def _lpi(point, space, model, n):
    """Local parameter importance for each hyperparameters"""
    grid = make_grid(point, space, model, n)
    variances = compute_variances(grid)
    ratios = variances / variances.sum()
    return pd.DataFrame(data=ratios, index=space.keys(), columns=["LPI"])


def _linear_lpi(point, space, model, n):
    # TODO
    return


modes = dict(best=_lpi, linear=_linear_lpi)


def lpi(trials, space, mode="best", model="RandomForestRegressor", n=20, **kwargs):
    """
    Calculates the Local Parameter Importance for a collection of :class:`Trial`.

    For more information on the metric, see original paper at
    https://ml.informatik.uni-freiburg.de/papers/18-LION12-CAVE.pdf.

    Biedenkapp, André, et al. "Cave: Configuration assessment, visualization and evaluation."
    International Conference on Learning and Intelligent Optimization. Springer, Cham, 2018.

    Parameters
    ----------
    trials: DataFrame or dict
        A dataframe of trials containing, at least, the columns 'objective' and 'id'. Or a dict
        equivalent.

    space: Space object
        A space object from an experiment.

    mode: str
        Mode to compute the LPI.
        - ``best``: Take the best trial found as the anchor for the LPI
        - ``linear``: Recompute LPI for all values on a grid

    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)

    n: int
        Number of points to compute the variances. Default is 20.

    **kwargs
        Arguments for the regressor model.

    Returns
    -------
    DataFrame
        LPI value for each parameter. If ``mode`` is `linear`, then a list of
        param values and LPI metrics are returned in a DataFrame format.
    """
    flattened_space = build_required_space(
        space, type_requirement="numerical", shape_requirement="flattened"
    )
    if trials.empty or trials.shape[0] == 0:
        return pd.DataFrame(
            data=[0] * len(flattened_space),
            index=flattened_space.keys(),
            columns=["LPI"],
        )

    data = to_numpy(trials, space)
    data = flatten(data, flattened_space)
    model = train_regressor(model, data, **kwargs)
    best_point = data[numpy.argmin(data[:, -1])]
    results = modes[mode](best_point, flattened_space, model, n)
    return results
