# -*- coding: utf-8 -*-
"""
Base Executor
=============

Base executor class for the parallelisation of experiments.

"""

from orion.core.utils import Factory


class BaseExecutor:
    """Base executor class

    Parameters
    ----------
    n_workers: int
        The number of workers the Executor should have. Depending on the backend
        it may spawn this many worker or connect to running instance that
        dispatch work to ``n_workers`` workers.

    """

    def __init__(self, n_workers, **kwargs):
        self.n_workers = n_workers

    def __getstate__(self):
        return dict(n_workers=self.n_workers)

    def __setstate__(self, state):
        self.n_workers = state["n_workers"]

    def wait(self, futures):
        """Wait for all futures to complete execution.

        Parameters
        ----------
        futures: `concurrent.futures.Futures` or equivalent interface
            The objects returned by ``submit()`` of the executor.

        """
        pass

    def submit(self, function, *args, **kwargs):
        """Submit work to the executor for asynchronous execution

        Parameters
        ----------
        function: a callable object
            A function to be executed by the executor. The function must be serializable.
        *args, **kwargs:
            Arguments for the function. The arguments must be serializable.

        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# pylint: disable=too-few-public-methods,abstract-method
class Executor(BaseExecutor, metaclass=Factory):
    """Factory class to build Executors"""
