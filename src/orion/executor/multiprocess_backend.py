import dataclasses
from multiprocessing import Pool, Manager
from multiprocessing.pool import AsyncResult
from queue import Empty
import uuid
from dataclasses import dataclass

from orion.executor.base import BaseExecutor


class Multiprocess(BaseExecutor):
    """Simple multiprocess executor that wraps ``multiprocessing.Pool``."""

    def __init__(self, n_workers, **kwargs):
        super().__init__(n_workers, **kwargs)
        self.manager = Manager()
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
        return self.pool.apply_async(function, args, kwargs)

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
                    tobe_deleted = [future]
                    break

        for future in tobe_deleted:
            futures.remove(future)

        if tobe_raised:
            raise tobe_raised

        return results
