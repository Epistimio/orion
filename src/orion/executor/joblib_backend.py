import joblib

from orion.executor.base import BaseExecutor


class Joblib(BaseExecutor):
    def __init__(self, n_workers=-1, backend="loky", **config):
        super(Joblib, self).__init__(n_workers=n_workers)
        self.backend = backend
        self.config = config

        self.joblib_parallel = joblib.parallel_backend(
            self.backend, n_jobs=self.n_workers, **self.config
        )

    def __getstate__(self):
        state = super(Joblib, self).__getstate__()
        state["backend"] = self.backend
        state["config"] = self.config
        return state

    def __setstate__(self, state):
        super(Joblib, self).__setstate__(state)
        self.backend = state["backend"]
        self.config = state["config"]

        self.joblib_parallel = joblib.parallel_backend(
            self.backend, n_jobs=self.n_workers, **self.config
        )

    def wait(self, futures):
        return joblib.Parallel(n_jobs=self.n_workers)(futures)

    def submit(self, function, *args, **kwargs):
        return joblib.delayed(function)(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.joblib_parallel.unregister()
        super(Joblib, self).__exit__(exc_type, exc_value, traceback)
