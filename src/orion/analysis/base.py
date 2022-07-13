"""
Base tools to compute diverse analysis
======================================

"""
import itertools
from functools import reduce

import numpy
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from orion.core.utils import format_trials
from orion.core.worker.transformer import build_required_space

_regressors_ = {
    "AdaBoostRegressor": AdaBoostRegressor,
    "BaggingRegressor": BaggingRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}


# pylint:disable=dangerous-default-value
def average(trials, group_by="order", key="best", return_var=False):
    """Compute the average of some trial attribute.

    By default it will compute the average objective at each time step across
    multiple experiments.

    Parameters
    ----------
    trials: DataFrame
        A dataframe of trials containing, at least, the columns 'best' and 'order'.
    group_by: str, optional
        The attribute to use to group trials for the average. By default it group trials
        by order (ex: all first trials across experiments.)
    key: str, optional
        One attribute or a list of attributes split by ',' to average.
        Defaults to 'best' as returned by ``orion.analysis.regret``.
    return_var: bool, optional
        If True, and a column '{key}_var' where '{key}' is the value of the argument `key`.
        Defaults to False.

    Returns
    -------
    A dataframe with columns 'order', '{key}_mean' and '{key}_var'.

    """
    if trials.empty:
        return trials

    group = trials.groupby(group_by)
    means = []
    keys = [v.strip() for v in key.split(",")]
    for k in keys:
        mean = group[k].mean().reset_index().rename(columns={k: f"{k}_mean"})
        if return_var:
            mean[f"{k}_var"] = group[k].var().reset_index()[k]
        means.append(mean)

    df_merged = reduce(pd.merge, means)

    return df_merged


# pylint:disable=unsupported-assignment-operation
def ranking(trials, group_by="order", key="best"):
    """Compute the ranking of some trial attribute.

    By default it will compute the ranking with respect to objectives at each time step across
    multiple experiments.

    Parameters
    ----------
    trials: DataFrame
        A dataframe of trials containing, at least, the columns 'best' and 'order'.
    group_by: str, optional
        The attribute to use to group trials for the ranking. By default it group trials
        by order (ex: all first trials across experiments.)
    key: str, optional
        The attribute to use for the ranking. Defaults to 'best' as returned by
        ``orion.analysis.regret``.

    Returns
    -------
    A copy of the original dataframe with a new column 'rank' for the rankings.

    """
    if trials.empty:
        return trials

    def rank(row):
        indices = row[key].argsort().to_numpy()
        ranks = numpy.empty_like(indices)
        ranks[indices] = numpy.arange(len(ranks))
        row["rank"] = ranks
        return row

    return trials.groupby(group_by).apply(rank)


def flatten_params(space, params=None):
    """Return the params of the corresponding flat space

    If no params are passed, returns all flattened params.
    If params are passed, returns the corresponding flattened params.

    Parameters
    ----------
    space: Space object
        A space object from an experiment.
    params: list of str, optional
        The parameters to select from the search space. If the flattened search space
        contains flattened params such as ('y' -> 'y[0]', 'y[1]'), passing 'y' in the list of
        params will returned the flattened version ['y[0]', 'y[1]']

    Examples
    --------
    If space has x~uniform(0, 1) and y~uniform(0, 1, shape=(1, 2)).
    >>> flatten_params(space)
    ['x', 'y[0,0]', 'y[0,1]']
    >>> flatten_params(space, params=['x'])
    ['x']
    >>> flatten_params(space, params=['x', 'y'])
    ['x', 'y[0,0]', 'y[0,1]']
    >>> flatten_params(space, params=['x', 'y[0,1]'])
    ['x', 'y[0,1]']
    >>> flatten_params(space, params=['y[0,1]', 'x'])
    ['x', 'y[0,1]']

    Raises
    ------
    ValueError
        If one of the parameter names passed is not in the flattened space.

    """
    keys = set(space.keys())
    flattened_keys = set(
        build_required_space(
            space,
            dist_requirement="linear",
            type_requirement="numerical",
            shape_requirement="flattened",
        ).keys()
    )

    if params is None:
        return sorted(flattened_keys)

    flattened_params = []
    for param in params:
        if param not in flattened_keys and param not in keys:
            raise ValueError(
                f"Parameter {param} not contained in space: {flattened_keys}"
            )
        elif param not in flattened_keys and param in keys:
            dim = space[param]
            flattened_params += [
                f'{dim.name}[{",".join(map(str, index))}]'
                for index in itertools.product(*map(range, dim.shape))
            ]
        else:
            flattened_params.append(param)

    return flattened_params


def to_numpy(trials, space):
    """Convert trials in DataFrame to Numpy array of (params + objective)"""
    return trials[list(space.keys()) + ["objective"]].to_numpy()


def flatten_numpy(trials_array, flattened_space):
    """Flatten dimensions"""

    flattened_points = numpy.array(
        [
            format_trials.trial_to_tuple(
                flattened_space.transform(
                    format_trials.tuple_to_trial(point[:-1], flattened_space.original)
                ),
                flattened_space,
            )
            for point in trials_array
        ]
    )

    return numpy.concatenate((flattened_points, trials_array[:, -1:]), axis=1)


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
            f"Did you mean any of thesis: list(_regressors_.keys())"
        )

    regressor = _regressors_[regressor_name](**kwargs)
    return regressor.fit(data[:, :-1], data[:, -1])
