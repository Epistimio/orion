import warnings

import joblib

from orion.executor.multiprocess_backend import Multiprocess


class Joblib(Multiprocess):
    def __init__(self, n_workers=-1, backend="loky", **config):
        warnings.warn(
            "Joblib backend is deprecated, use Multiprocess instead", DeprecationWarning
        )
        super(Joblib, self).__init__(n_workers=n_workers)
