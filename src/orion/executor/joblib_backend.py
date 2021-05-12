from orion.executor.base import BaseExecutor

import joblib


class Joblib(BaseExecutor):
    def __init__(self, experiment=None, n_jobs=-1, backend="loky", **config):
        super(Joblib, self).__init__(experiment, n_jobs)
        self.backend = backend
        self.config = config

        self.joblib_parallel = joblib.parallel_backend(
            self.backend, n_jobs=self.n_jobs, **self.config
        )
        self.parallel = joblib.Parallel(n_jobs=self.n_jobs)

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
            self.backend, n_jobs=self.n_jobs, **self.config
        )
        self.parallel = joblib.Parallel(n_jobs=self.n_jobs)

    def wait(self, futures):
        return self.parallel(futures)

    def submit(self, function, *args, **kwargs):
        return joblib.delayed(function)(*args, **kwargs)

    def __enter__(self):
        # Wrap storage with dask parallel wrapper and set storage of experiment
        super(Joblib, self).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # Add arguments
        # Set storage of experiment back to Oríon storage, get rid of dask wrapper
        self.joblib_parallel.unregister()
        super(Joblib, self).__exit__(exc_type, exc_value, traceback)
