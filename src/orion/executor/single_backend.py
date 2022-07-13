"""
Executor without parallelism for debugging
==========================================

"""
import functools
import time
import traceback

from orion.executor.base import (
    AsyncException,
    AsyncResult,
    BaseExecutor,
    ExecutorClosed,
    Future,
)

# A function can return None so we have to create a difference between
# the None result and the absence of result
NOT_SET = object()


class _Future(Future):
    """Wraps a partial function to act as a Future"""

    def __init__(self, future):
        self.future = future
        self.result = NOT_SET
        self.exception = NOT_SET

    def get(self, timeout=None):
        start = time.time()
        self.wait(timeout)

        if timeout and time.time() - start > timeout:
            raise TimeoutError()

        if self.result is not NOT_SET:
            return self.result

        else:
            raise self.exception

    def wait(self, timeout=None):
        if self.ready():
            return

        try:
            self.result = self.future()
        except Exception as e:
            self.exception = e

    def ready(self):
        return (self.result is not NOT_SET) or (self.exception is not NOT_SET)

    def successful(self):
        if not self.ready():
            raise ValueError()

        return self.exception is NOT_SET


class SingleExecutor(BaseExecutor):
    """Single thread executor

    Simple executor for debugging. No parameters.

    The submitted functions are wrapped with ``functools.partial``
    which are then executed in ``wait()``.

    Notes
    -----
    The tasks are started when wait is called

    """

    def __init__(self, n_workers=1, **config):
        super().__init__(n_workers=1)
        self.closed = False
        self.nested = 0

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()

    def __enter__(self):
        self.nested += 1
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        """Prevent user from submitting work after closing."""
        if self.nested <= 1:
            self.closed = True

    def wait(self, futures):
        return [future.get() for future in futures]

    def async_get(self, futures, timeout=0.01):
        if len(futures) == 0:
            return []

        results = []
        try:
            fut = futures.pop()
            results.append(AsyncResult(fut, fut.get()))
        except Exception as err:
            results.append(AsyncException(fut, err, traceback.format_exc()))

        return results

    def submit(self, function, *args, **kwargs):
        if self.closed:
            raise ExecutorClosed()

        return _Future(functools.partial(function, *args, **kwargs))
