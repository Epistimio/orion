import cloudpickle

import pickle
import dataclasses
from dataclasses import dataclass
import uuid
from multiprocessing import Pool, Manager
from multiprocessing.pool import AsyncResult
from queue import Empty

from orion.executor.base import BaseExecutor


def _couldpickle_exec(payload):
    function, args, kwargs = pickle.loads(payload)
    result = function(*args, **kwargs)
    return cloudpickle.dumps(result)


class _Future:
    """Wraps a python AsyncResult"""

    def __init__(self, future):
        self.future = future

    def get(self, timeout=None):
        r = self.future.get(timeout)
        return pickle.loads(r)

    def wait(self, timeout=None):
        return self.future.wait(timeout)

    def ready(self):
        return self.future.ready()

    def succesful(self):
        return self.future.succesful()


class Multiprocess(BaseExecutor):
    """Simple multiprocess executor that wraps ``multiprocessing.Pool``."""

    def __init__(self, n_workers, **kwargs):
        super().__init__(n_workers, **kwargs)
        self.pool = Pool(n_workers)

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
        payload = cloudpickle.dumps((function, args, kwargs))
        return _Future(self.pool.apply_async(_couldpickle_exec, (payload,)))

    def wait(self, futures):
        return [
            future.get() for future in futures
        ]

    def _find_future_exception(self, future):
        for _, status in self.futures.items():
            if id(status.future) == id(future):
                return status.exception

    def async_get(self, futures, timeout=None):
        results = []
        tobe_deleted = []
        tobe_raised = None

        for future in futures:
            if timeout:
                future.wait(timeout)

            if future.ready():
                try:
                    results.append(future.get())
                    tobe_deleted.append(future)

                except Exception as err:
                    # delay the raising of the exception so we are allowed to remove
                    # the future that raised it
                    # it means once it is handled waitone will proceed as expected
                    tobe_raised = err
                    results = []
                    tobe_deleted = [future]
                    break

        for future in tobe_deleted:
            futures.remove(future)

        if tobe_raised:
            raise tobe_raised

        return results
