import concurrent.futures
import logging
import multiprocessing
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import Process
from multiprocessing.pool import Pool as PyPool

import cloudpickle

from orion.executor.base import (
    AsyncException,
    AsyncResult,
    BaseExecutor,
    ExecutorClosed,
    Future,
)

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
    """Wraps a python AsyncResult and pickle the payload using cloudpickle
    to enable the use of more python objects as functions and arguments,
    which makes the multiprocess backend on par with Dask.

    """

    def __init__(self, future, cloudpickle=False):
        self.future = future
        self.cloudpickle = cloudpickle

    def get(self, timeout=None):
        try:
            r = self.future.get(timeout)
            return pickle.loads(r) if self.cloudpickle else r
        except multiprocessing.context.TimeoutError as e:
            raise TimeoutError() from e

    def wait(self, timeout=None):
        return self.future.wait(timeout)

    def ready(self):
        return self.future.ready()

    def successful(self):
        # Python 3.6 raise assertion error
        if not self.ready():
            raise ValueError()

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
        try:
            return self.future.result(timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError() from e

    def wait(self, timeout=None):
        wait([self.future], timeout)

    def ready(self):
        return self.future.done()

    def successful(self):
        if not self.future.done():
            raise ValueError()

        return self.future.exception() is None


class ThreadPool:
    """Custom pool that creates multiple threads instead of processes"""

    def __init__(self, n_workers):
        self.pool = ThreadPoolExecutor(n_workers)

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

    .. warning::

       Pickling of the executor is not supported, see Dask for a backend that supports it

    """

    BACKENDS = dict(
        thread=ThreadPool,
        threading=ThreadPool,
        multiprocess=Pool,
        loky=Pool,  # TODO: For compatibility with joblib backend. Remove in v0.4.0.
    )

    def __init__(self, n_workers=-1, backend="multiprocess", **kwargs):
        super().__init__(n_workers, **kwargs)

        if n_workers <= 0:
            n_workers = multiprocessing.cpu_count()

        self.pool = PoolExecutor.BACKENDS.get(backend, ThreadPool)(n_workers)

    def __setstate__(self, state):
        self.pool = state["pool"]

    def __getstate__(self):
        return dict(pool=self.pool)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        # This is necessary because if the factory constructor fails
        # __del__ is executed right away but pool might not be set
        if hasattr(self, "pool"):
            self.pool.shutdown()

    def submit(self, function, *args, **kwargs):
        try:
            return self._submit_cloudpickle(function, *args, **kwargs)
        except ValueError as e:
            if str(e).startswith("Pool not running"):
                raise ExecutorClosed() from e

            raise
        except RuntimeError as e:
            if str(e).startswith("cannot schedule new futures after shutdown"):
                raise ExecutorClosed() from e

            raise

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
