"""
Base Executor
=============

Base executor class for the parallelisation of experiments.

"""

from orion.core.utils import GenericFactory


class ExecutorClosed(Exception):
    """Raised when submitting to a closed executor"""


class AsyncResult:
    """Result of an async computation"""

    def __init__(self, future, v):
        self.future = future
        self.value = v


class AsyncException:
    """Exception raised by a remote worker during computation"""

    def __init__(self, future, exception, traceback):
        self.future = future
        self.exception = exception
        self.traceback = traceback

    @property
    def value(self):
        """Raise the exception"""
        raise self.exception


class Future:
    """Generic Future interface that is used to harmonized different future interface"""

    def get(self, timeout=None):
        """Return the result when it arrives.
        If the remote call raised an exception then that exception will be reraised by get().

        Parameters
        ----------
        timeout: int
            time in second to wait, if none will wait forever

        Raises
        ------
        multiprocessing.TimeoutError
            when the timeout expires

        Exception
            if the remote called raised an exception

        """

    def wait(self, timeout=None):
        """Wait until the result is available or until timeout seconds pass."""

    def ready(self):
        """Return whether the call has completed."""

    def successful(self):
        """Return whether the call completed without raising an exception.
        Will raise ValueError if the result is not ready.

        Raises
        ------
        ValueError
            if the result is not yet ready

        """


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

    def async_get(self, futures, timeout=None):
        """Retrieve futures that completed, removes them from the list of pending futures
        and return their results

        Parameters
        ----------
        futures: `concurrent.futures.Futures` or equivalent interface
            The objects returned by ``submit()`` of the executor.

        timeout: int
            time to wait before checking the other future

        Returns
        -------
        returns a list of results

        """

    def submit(self, function, *args, **kwargs):
        """Submit work to the executor for asynchronous execution

        Parameters
        ----------
        function: a callable object
            A function to be executed by the executor. The function must be serializable.
        *args, **kwargs:
            Arguments for the function. The arguments must be serializable.

        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


executor_factory = GenericFactory(BaseExecutor)
