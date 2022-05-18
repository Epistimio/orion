import warnings

from orion.executor.multiprocess_backend import PoolExecutor


class Joblib(PoolExecutor):
    def __init__(self, n_workers=-1, backend="loky", **config):
        warnings.warn(
            "Joblib is deprecated and will be removed in v0.4.0."
            "Use PoolExecutor instead.",
            DeprecationWarning,
        )
        super().__init__(n_workers=n_workers, backend=backend)
