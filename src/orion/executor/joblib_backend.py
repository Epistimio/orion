import traceback

import joblib

from orion.executor.base import BaseExecutor
from orion.executor.base import AsyncException, AsyncResult, BaseExecutor


class _Future:
    """Wraps a python AsyncResult"""

    def __init__(self, fun, *args, **kwargs):
        self.future = self
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def get(self, timeout=None):
        return None

    def wait(self, timeout=None):
        return None

    def ready(self):
        return False

    def succesful(self):
        raise ValueError()

    def delayed(self):
        return joblib.delayed(self.fun)(*self.args, **self.kwargs)


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

    def async_get(self, futures, timeout=None):
        return self.wait(futures)

    def wait(self, futures):
        try:
            results = joblib.Parallel(n_jobs=self.n_workers)(
                [f.delayed() for f in futures]
            )

            async_results = []
            for future, result in zip(futures, results):
                async_results.append(AsyncResult(future, result))

            return async_results
        except Exception as e:
            return [AsyncException(None, e, traceback.format_exc())]

    def submit(self, function, *args, **kwargs):
        return _Future(function, *args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.joblib_parallel.unregister()
        super(Joblib, self).__exit__(exc_type, exc_value, traceback)
