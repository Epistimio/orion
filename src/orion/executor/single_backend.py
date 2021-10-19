# -*- coding: utf-8 -*-
"""
Executor without parallelism for debugging
==========================================

"""
import functools

from orion.executor.base import BaseExecutor


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
        return [future() for future in futures]

    def waitone(self, futures, timeout=0.01):
        if len(futures) == 0:
            return []

        fut = futures()
        result = fut()
        futures.remove(fut)

        return [result]

    def submit(self, function, *args, **kwargs):
        return functools.partial(function, *args, **kwargs)
