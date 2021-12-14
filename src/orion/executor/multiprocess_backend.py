import dataclasses
import logging
import pickle
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait
from dataclasses import dataclass
from multiprocessing import Manager, Process, get_context
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import Pool as PyPool
from queue import Empty

import cloudpickle

from orion.executor.base import AsyncException, AsyncResult, BaseExecutor, Future

log = logging.getLogger(__name__)


def _couldpickle_exec(payload):
    function, args, kwargs = pickle.loads(payload)
    result = function(*args, **kwargs)
    return cloudpickle.dumps(result)


class _Process(Process):
    """Process that cannot be a daemon"""

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class _Future(Future):
    """Wraps a python AsyncResult"""

    def __init__(self, future, cloudpickle=False):
        self.future = future
        self.cloudpickle = cloudpickle

    def get(self, timeout=None):
        r = self.future.get(timeout)
        return pickle.loads(r) if self.cloudpickle else r

    def wait(self, timeout=None):
        return self.future.wait(timeout)

    def ready(self):
        return self.future.ready()

    def successful(self):
        return self.future.successful()


class Pool(PyPool):
    """Custom pool that does not set its worker as daemon process"""

    ALLOW_DAEMON = True

    @staticmethod
    def Process(*args, **kwds):
        import sys

        v = sys.version_info

        #  < 3.8 use self._ctx
        # >= 3.8 ctx as an argument
        if v.major == 3 and v.minor >= 8:
            args = args[1:]

        if Pool.ALLOW_DAEMON:
            return Process(*args, **kwds)

        return _Process(*args, **kwds)

    def shutdown(self):
        # NB: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
        # says to not use terminate although it is what __exit__ does
        self.close()
        self.join()


class _ThreadFuture(Future):
    """Wraps a concurrent Future to behave like AsyncResult"""

    def __init__(self, future):
        self.future = future

    def get(self, timeout=None):
        return self.future.result(timeout)

    def wait(self, timeout=None):
        wait([self.future], timeout)

    def ready(self):
        return self.future.done()

    def successful(self):
        if not self.future.done():
            raise ValueError()

        return self.future.exception() is None


class ThreadPool:
    """Custom pool that creates multiple threads instead of processess"""

    def __init__(self, n_workers):
        self.pool = ThreadPoolExecutor(n_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.shutdown()

    def shutdown(self):
        self.pool.shutdown()

    def apply_async(self, fun, args, kwds=None):
        if kwds is None:
            kwds = dict()

        return _ThreadFuture(self.pool.submit(fun, *args, **kwds))


class PoolExecutor(BaseExecutor):
    """Simple Pool executor.

    Parameters
    ----------

    n_workers: int
        Number of workers to spawn

    backend: str
        Pool backend to use; thread or multiprocess, defaults to multiprocess

    """

    BACKENDS = dict(
        thread=ThreadPool,
        threading=ThreadPool,
        multiprocess=Pool,
        loky=Pool,  # TODO: For compatibility with joblib backend. Remove in v0.4.0.
    )

    def __init__(self, n_workers, backend="multiprocess", **kwargs):
        self.pool = PoolExecutor.BACKENDS.get(backend, ThreadPool)(n_workers)
        super().__init__(n_workers, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.shutdown()

    def __del__(self):
        self.pool.shutdown()

    def __getstate__(self):
        state = super(PoolExecutor, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(PoolExecutor, self).__setstate__(state)

    def submit(self, function, *args, **kwargs):
        return self._submit_cloudpickle(function, *args, **kwargs)

    def _submit_cloudpickle(self, function, *args, **kwargs):
        payload = cloudpickle.dumps((function, args, kwargs))
        return _Future(self.pool.apply_async(_couldpickle_exec, args=(payload,)), True)

    def wait(self, futures):
        return [future.get() for future in futures]

    def async_get(self, futures, timeout=None):
        results = []
        tobe_deleted = []

        for i, future in enumerate(futures):
            if timeout and i == 0:
                future.wait(timeout)

            if future.ready():
                try:
                    results.append(AsyncResult(future, future.get()))
                except Exception as err:
                    results.append(AsyncException(future, err, traceback.format_exc()))

                tobe_deleted.append(future)

        for future in tobe_deleted:
            futures.remove(future)

        return results
