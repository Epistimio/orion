# -*- coding: utf-8 -*-
"""
Executor without parallelism for debugging
==========================================

"""
import functools
import traceback

from orion.executor.base import AsyncException, AsyncResult, BaseExecutor


class _Future:
    """Wraps a partial function to act as a Future"""

    def __init__(self, future):
        self.future = future
        self.result = None
        self.exception = None

    def get(self, timeout=None):
        if self.result:
            return self.result

        if self.exception:
            raise self.exception

        self.result = self.future()
        return self.result

    def wait(self, timeout=None):
        try:
            self.result = self.future()
        except Exception as e:
            self.exception = e

    def ready(self):
        return self.result is None

    def successful(self):
        if self.result is None:
            raise ValueError()

        return self.exception is None


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
        super(SingleExecutor, self).__init__(n_workers=1)

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
        return _Future(functools.partial(function, *args, **kwargs))
