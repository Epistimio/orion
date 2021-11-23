import dataclasses
import logging
import pickle
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from multiprocessing import Manager, Process
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import Pool as PyPool
from queue import Empty

import cloudpickle

from orion.executor.base import AsyncException, AsyncResult, BaseExecutor

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


class _Future:
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

    @staticmethod
    def Process(*args, **kwds):
        import sys

        v = sys.version_info

        #  < 3.8 use self._ctx
        # >= 3.8 ctx as an argument
        if v.major == 3 and v.minor >= 8:
            args = args[1:]

        return _Process(*args, **kwds)


class _ThreadFuture:
    """Wraps a concurrent Future to behave like AsyncResult"""

    def __init__(self, future):
        self.future = future
        self.result = None

    def get(self, timeout=None):
        if self.result:
            return self.result

        self.result = self.future.result(timeout)
        return self.result

    def wait(self, timeout=None):
        try:
            self.result = self.future.result(timeout)
        except TimeoutError:
            pass

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

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.shutdown()

    def terminate(self):
        self.pool.shutdown()

    def apply_async(self, fun, args, kwds=None):
        if kwds is None:
            kwds = dict()

        return _ThreadFuture(self.pool.submit(fun, *args, **kwds))


class Multiprocess(BaseExecutor):
    """Simple multiprocess executor that wraps ``multiprocessing.Pool``."""

    def __init__(self, n_workers, **kwargs):
        super().__init__(n_workers, **kwargs)
        # self.pool = Pool(n_workers)
        self.pool = ThreadPool(n_workers)

    def __del__(self):
        self.pool.terminate()

    def __getstate__(self):
        state = super(Multiprocess, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(Multiprocess, self).__setstate__(state)

    def __enter__(self):
        self.pool.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)

    def submit(self, function, *args, **kwargs) -> AsyncResult:
        return self._submit_cloudpickle(function, *args, **kwargs)

    def _submit_python(self, function, *args, **kwargs) -> AsyncResult:
        return _Future(self.pool.apply_async(function, args=args, kwds=kwargs))

    def _submit_cloudpickle(self, function, *args, **kwargs) -> AsyncResult:
        payload = cloudpickle.dumps((function, args, kwargs))
        return _Future(self.pool.apply_async(_couldpickle_exec, args=(payload,)), True)

    def wait(self, futures):
        return [future.get() for future in futures]

    def _find_future_exception(self, future):
        for _, status in self.futures.items():
            if id(status.future) == id(future):
                return status.exception

    def async_get(self, futures, timeout=None):
        results = []
        tobe_deleted = []

        for future in futures:
            if timeout:
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
