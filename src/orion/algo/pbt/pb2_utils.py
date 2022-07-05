# pylint: skip-file
"""PB2 Utils from Ray tune package, used in PB2 explore method.

Reference (2022/02/18):
https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pb2_utils.py
"""
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import euclidean_distances, pairwise_distances

from orion.core.utils.module_import import ImportOptional

with ImportOptional("PB2") as import_optional:
    import GPy
    from GPy import Param
    from GPy.kern import Kern

if import_optional.failed:
    GPy = None
    Param = None

    class Kern:
        def __init__(self, *args, **kwargs):
            pass


class TVSquaredExp(Kern):
    """Time varying squared exponential kernel.
    For more info see the TV-GP-UCB paper:
    http://proceedings.mlr.press/v51/bogunovic16.pdf
    """

    def __init__(
        self, input_dim, variance=1.0, lengthscale=1.0, epsilon=0.0, active_dims=None
    ):
        import_optional.ensure()
        super().__init__(input_dim, active_dims, "time_se")
        self.variance = Param("variance", variance)
        self.lengthscale = Param("lengthscale", lengthscale)
        self.epsilon = Param("epsilon", epsilon)
        self.link_parameters(self.variance, self.lengthscale, self.epsilon)

    def K(self, X, X2):
        # time must be in the far left column
        if self.epsilon > 0.5:  # 0.5
            self.epsilon = 0.5
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel = (1 - self.epsilon) ** (0.5 * dists)

        X = X[:, 1:]
        X2 = X2[:, 1:]

        RBF = self.variance * np.exp(
            -np.square(euclidean_distances(X, X2)) / self.lengthscale
        )

        return RBF * timekernel

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        dist2 = np.square(euclidean_distances(X, X2)) / self.lengthscale

        dvar = np.exp(-np.square((euclidean_distances(X, X2)) / self.lengthscale))
        dl = -(
            2 * euclidean_distances(X, X2) ** 2 * self.variance * np.exp(-dist2)
        ) * self.lengthscale ** (-2)
        n = pairwise_distances(T1, T2, "cityblock") / 2
        deps = -n * (1 - self.epsilon) ** (n - 1)

        self.variance.gradient = np.sum(dvar * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)
        self.epsilon.gradient = np.sum(deps * dL_dK)


def normalize(data, wrt):
    """Normalize data to be in range (0,1), with respect to (wrt) boundaries,
    which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (
        np.max(wrt, axis=0) - np.min(wrt, axis=0) + 1e-8
    )


def standardize(data):
    """Standardize to be Gaussian N(0,1). Clip final values."""
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return np.clip(data, -2, 2)


def UCB(m, m1, x, fixed, kappa=0.5):
    """UCB acquisition function.

    Interesting points to note:

    1. We concat with the fixed points, because we are not optimizing wrt
       these. This is the Reward and Time, which we can't change. We want
       to find the best hyperparameters *given* the reward and time.
    2. We use m to get the mean and m1 to get the variance. If we already
       have trials running, then m1 contains this information. This reduces
       the variance at points currently running, even if we don't have
       their label.
       Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

    """

    c1 = 0.2
    c2 = 0.4
    beta_t = c1 * np.log(c2 * m.X.shape[0])
    kappa = np.sqrt(beta_t)

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T

    try:
        preds = m.predict(xtest)
        preds = m.predict(xtest)
        mean = preds[0][0][0]
    except ValueError:
        mean = -9999

    try:
        preds = m1.predict(xtest)
        var = preds[1][0][0]
    except ValueError:
        var = 0
    return mean + kappa * var


def optimize_acq(func, m, m1, fixed, num_f):
    """Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = m1.X[0, :]

    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)

        res = minimize(
            lambda x: -func(m, m1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )

        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return np.clip(best_theta, 0, 1)


def select_length(Xraw, yraw, bounds, num_f):
    """Select the number of datapoints to keep, using cross validation"""
    min_len = 200

    if Xraw.shape[0] < min_len:
        return Xraw.shape[0]
    else:
        length = min_len - 10
        scores = []
        while length + 10 <= Xraw.shape[0]:
            length += 10

            base_vals = np.array(list(bounds.values())).T
            X_len = Xraw[-length:, :]
            y_len = yraw[-length:]
            oldpoints = X_len[:, :num_f]
            old_lims = np.concatenate(
                (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
            ).reshape(2, oldpoints.shape[1])
            limits = np.concatenate((old_lims, base_vals), axis=1)

            X = normalize(X_len, limits)
            y = standardize(y_len).reshape(y_len.size, 1)

            kernel = TVSquaredExp(
                input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
            )
            m = GPy.models.GPRegression(X, y, kernel)
            m.optimize(messages=True)

            scores.append(m.log_likelihood())
        idx = np.argmax(scores)
        length = (idx + int((min_len / 10))) * 10
        return length


def select_config(Xraw, yraw, current, newpoint, bounds, num_f):
    """Selects the next hyperparameter config to try.

    This function takes the formatted data, fits the GP model and optimizes the
    UCB acquisition function to select the next point.

    Parameters
    ----------
    Xraw: np.array
        The un-normalized array of hyperparams, Time and Reward
    yraw: np.array
        The un-normalized vector of reward changes.
    current: list
        The hyperparams of trials currently running. This is important so we do not select the same
        config twice. If there is data here then we fit a second GP including it
        (with fake y labels). The GP variance doesn't depend on the y labels so it is ok.
    newpoint: np.array
        The Reward and Time for the new point.  We cannot change these as they are based on the
        *new weights*.
    bounds: dict
        Bounds for the hyperparameters. Used to normalize.
    num_f: int
        The number of fixed params. Almost always 2 (reward+time)

    Returns
    -------
    ``np.array``
        A vector of new hyperparameters.
    """
    length = select_length(Xraw, yraw, bounds, num_f)

    Xraw = Xraw[-length:, :]
    yraw = yraw[-length:]

    base_vals = np.array(list(bounds.values())).T
    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate(
        (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    ).reshape(2, oldpoints.shape[1])
    limits = np.concatenate((old_lims, base_vals), axis=1)

    X = normalize(Xraw, limits)
    y = standardize(yraw).reshape(yraw.size, 1)

    fixed = normalize(newpoint, oldpoints)

    kernel = TVSquaredExp(
        input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
    )

    try:
        m = GPy.models.GPRegression(X, y, kernel)
    except np.linalg.LinAlgError:
        # add diagonal ** we would ideally make this something more robust...
        X += np.eye(X.shape[0]) * 1e-3
        m = GPy.models.GPRegression(X, y, kernel)

    try:
        m.optimize()
    except np.linalg.LinAlgError:
        # add diagonal ** we would ideally make this something more robust...
        X += np.eye(X.shape[0]) * 1e-3
        m = GPy.models.GPRegression(X, y, kernel)
        m.optimize()

    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))

    if current is None:
        m1 = deepcopy(m)
    else:
        # add the current trials to the dataset
        padding = np.array([fixed for _ in range(current.shape[0])])
        current = normalize(current, base_vals)
        current = np.hstack((padding, current))

        Xnew = np.vstack((X, current))
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1, 1)
        ynew = np.vstack((y, ypad))

        # kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1.,
        # lengthscale=1.)
        kernel = TVSquaredExp(
            input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
        )
        m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
        m1.optimize()

    xt = optimize_acq(UCB, m, m1, fixed, num_f)

    # convert back...
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
        base_vals, axis=0
    )

    xt = xt.astype(np.float32)
    return xt
