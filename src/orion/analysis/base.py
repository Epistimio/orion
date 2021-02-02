"""
:mod:`orion.analysis.base` -- Base tools to compute diverse analysis
====================================================================

Base tools to compute diverse analysis

"""
import itertools

import numpy
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
        [flattened_space.transform(point[:-1]) for point in trials_array]
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
            f"Did you mean any of theses: list(_regressors_.keys())"
        )

    regressor = _regressors_[regressor_name](**kwargs)
    return regressor.fit(data[:, :-1], data[:, -1])
