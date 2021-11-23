import warnings

import joblib

from orion.executor.multiprocess_backend import PoolExecutor


class Joblib(PoolExecutor):
    def __init__(self, n_workers=-1, backend="loky", **config):
        warnings.warn(
            "Joblib backend is deprecated, use PoolExecutor instead", DeprecationWarning
        )
        super(Joblib, self).__init__(n_workers=n_workers)
