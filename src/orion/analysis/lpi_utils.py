"""
Provide tools to calculate Local Parameter Importance
=====================================================
"""
import numpy
import pandas as pd

from orion.analysis.base import flatten_numpy, to_numpy, train_regressor
from orion.core.worker.transformer import build_required_space


def make_grid(point, space, model, n_points):
    """Build a grid based on point.

    The shape of the grid will be
        (number of hyperparameters,
         number of points ``n_points``,
         number of hyperparameters + 1)

    Last column is the objective predicted by the model for a given point.

    Parameters
    ----------
    point: numpy.ndarray
        A tuple representation of the best trials, (hyperparameters + objective)
    space: Space object
        A space object from an experiment. It must be flattened and linearized.
    model: `sklearn.base.RegressorMixin`
        Trained regressor used to compute predictions on the grid
    n_points: int
        Number of points for each dimension on the grid.

    """
    grid = numpy.zeros((len(space), n_points, len(space) + 1))
    for i, dim in enumerate(space.values()):
        grid[i, :, :] = point
        grid[i, :, i] = numpy.linspace(*dim.interval(), num=n_points)
        grid[i, :, -1] = model.predict(grid[i, :, :-1])
    return grid


def compute_variances(grid):
    """Compute variance for each hyperparameters"""
    return grid[:, :, -1].var(axis=1)


def _lpi(point, space, model, n_points):
    """Local parameter importance for each hyperparameters"""
    grid = make_grid(point, space, model, n_points)
    variances = compute_variances(grid)
    ratios = variances / variances.sum()
    return ratios


# def _linear_lpi(point, space, model, n):
#     # TODO
#     return


modes = dict(best=_lpi)  # , linear=_linear_lpi)


def lpi(
    trials,
    space,
    mode="best",
    model="RandomForestRegressor",
    n_points=20,
    n_runs=10,
    **kwargs
):
    """
    Calculates the Local Parameter Importance for a collection of
    :class:`orion.core.worker.trial.Trial`.

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

    n_points: int
        Number of points to compute the variances. Default is 20.

    n_runs: int
        Number of runs to compute the standard error of the LPI. Default is 10.

    ``**kwargs``
        Arguments for the regressor model.

    Returns
    -------
    DataFrame
        LPI value for each parameter. If ``mode`` is `linear`, then a list of
        param values and LPI metrics are returned in a DataFrame format.

    """
    flattened_space = build_required_space(
        space,
        dist_requirement="linear",
        type_requirement="numerical",
        shape_requirement="flattened",
    )
    if trials.empty or trials.shape[0] == 0:
        return pd.DataFrame(
            data=[0] * len(flattened_space),
            index=flattened_space.keys(),
            columns=["LPI"],
        )

    data = to_numpy(trials, space)
    data = flatten_numpy(data, flattened_space)
    best_point = data[numpy.argmin(data[:, -1])]
    rng = numpy.random.RandomState(kwargs.pop("random_state", None))
    results = numpy.zeros((n_runs, len(flattened_space)))
    for i in range(n_runs):
        trained_model = train_regressor(
            model, data, random_state=rng.randint(2**32 - 1), **kwargs
        )
        results[i] = modes[mode](best_point, flattened_space, trained_model, n_points)

    averages = results.mean(0)
    standard_errors = results.std(0)
    frame = pd.DataFrame(
        data=numpy.array([averages, standard_errors]).T,
        index=flattened_space.keys(),
        columns=["LPI", "STD"],
    )
    return frame
