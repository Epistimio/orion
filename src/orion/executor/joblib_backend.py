import time

import joblib

from orion.executor.base import BaseExecutor


def _get_result(self, job, timeout):
    try:
        if getattr(self._backend, "supports_timeout", False):
            return job.get(timeout=timeout)
        else:
            return job.get()

    except BaseException as exception:
        # Note: we catch any BaseException instead of just Exception
        # instances to also include KeyboardInterrupt.

        # Stop dispatching any new job in the async callback thread
        self._aborting = True

        # If the backend allows it, cancel or kill remaining running
        # tasks without waiting for the results as we will raise
        # the exception we got back to the caller instead of returning
        # any result.
        backend = self._backend
        if backend is not None and hasattr(backend, "abort_everything"):
            # If the backend is managed externally we need to make sure
            # to leave it in a working state to allow for future jobs
            # scheduling.
            ensure_ready = self._managed_backend
            backend.abort_everything(ensure_ready=ensure_ready)
        raise


def retrieveone(self, timeout=0.01):
    results = []
    tobe_deleted = []
    self._output = []

    while self._iterating or len(self._jobs) > 0:
        for i, job in enumerate(self._jobs):
            result = _get_result(self, job, timeout)

            results.append((i, result[0]))
            tobe_deleted.append(job)
            self._output.append((i, result[0]))

        if results:
            break

    with self._lock:
        for job in tobe_deleted:
            self._jobs.remove(job)

    return results


class Joblib(BaseExecutor):
    """JobLib executor

    Notes
    -----
    The tasks are started when wait is called
    """

    def __init__(self, n_workers=-1, backend="loky", **config):
        super(Joblib, self).__init__(n_workers=n_workers)
        self.backend = backend
        self.config = config

        self.executor = None

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

    def _exec(self, futures):
        """Creates the joblib executor and returns the first set of results"""
        if self.executor is None:
            self.executor = joblib.Parallel(n_jobs=self.n_workers)
            self.executor.retrieve = lambda: retrieveone(self.executor)

        return self.executor(futures)

    def wait(self, futures):
        results = self._exec(futures)

        with self.executor._backend.retrieval_context():
            while len(self.executor._jobs) > 0:
                results.extend(retrieveone(self.executor))

        return results

    def waitone(self, futures):
        results = self._exec(futures)

        if results:
            return results

        with self.executor._backend.retrieval_context():
            return retrieveone(self.executor)

    def submit(self, function, *args, **kwargs):
        return joblib.delayed(function)(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.joblib_parallel.unregister()
        super(Joblib, self).__exit__(exc_type, exc_value, traceback)
