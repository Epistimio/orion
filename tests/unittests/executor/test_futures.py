import time

import pytest

from orion.executor.dask_backend import HAS_DASK, Dask
from orion.executor.multiprocess_backend import PoolExecutor
from orion.executor.single_backend import SingleExecutor


def multiprocess(n):
    """Create a Pool using the multiprocess backend"""
    return PoolExecutor(n, "multiprocess")


def thread(n):
    """Create a Pool using the threading backend"""
    return PoolExecutor(n, "threading")


backends = [
    thread,
    multiprocess,
    SingleExecutor,
    pytest.param(
        Dask,
        marks=pytest.mark.xfail(
            condition=not HAS_DASK, reason="Test requires dask.", raises=ImportError
        ),
    ),
]


class FunctionException(Exception):
    """Special exception for testing
    so we are sure we are catching the right exception

    """


def function(exception, sleep_time, result):
    """Simple test function"""
    time.sleep(sleep_time)
    if exception:
        raise exception
    return result


@pytest.mark.parametrize("backend", backends)
class TestFutures:
    """Test Future interface to make sure all backend behave the same"""

    def test_futures(self, backend):
        with backend(1) as exectuor:
            self.get_ok(exectuor)
            self.get_error(exectuor)
            self.get_timeout(exectuor)

            self.wait_ok(exectuor)
            self.wait_error(exectuor)
            self.wait_timeout(exectuor)

            self.ready_ok(exectuor)
            self.ready_error(exectuor)

            self.successful_ok(exectuor)
            self.successful_error(exectuor)

    def get_ok(self, executor):
        """Get - OK"""
        future = executor.submit(function, None, 0, 1)
        assert future.get() == 1

    def get_error(self, executor):
        """Get - Error"""

        future = executor.submit(function, FunctionException, 0, None)

        with pytest.raises(FunctionException):
            future.get()

    def get_timeout(self, executor):
        """Get - Timeout"""
        future = executor.submit(function, None, 1, 1)

        with pytest.raises(TimeoutError):
            future.get(0.01) == 1

    def wait_ok(self, executor):
        """Wait - OK"""
        future = executor.submit(function, None, 0.1, 1)

        assert future.ready() is False
        future.wait()
        assert future.ready() is True

    def wait_error(self, executor):
        """Wait - Error"""
        future = executor.submit(function, FunctionException, 0.1, None)

        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        with pytest.raises(FunctionException):
            future.get()

    def wait_timeout(self, executor):
        """Wait - Timeout"""
        future = executor.submit(function, FunctionException, 1, None)

        assert future.ready() is False

        future.wait(0.01)

        # SingleExecutor is not truly async so
        # results are always ready after a wait
        if not isinstance(executor, SingleExecutor):
            assert future.ready() is False
        else:
            assert future.ready() is True

        future.wait()
        assert future.ready() is True

    def ready_ok(self, executor):
        """Ready - OK"""
        future = executor.submit(function, None, 1, 1)
        assert future.ready() is False
        future.wait()
        assert future.ready() is True

    def ready_error(self, executor):
        """Ready - Error"""
        future = executor.submit(function, FunctionException, 0.1, None)
        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        with pytest.raises(FunctionException):
            future.get()

    def successful_ok(self, executor):
        """Successful - OK"""
        future = executor.submit(function, None, 1, 1)

        with pytest.raises(ValueError):
            assert future.successful()

        future.wait()
        assert future.successful() is True

    def successful_error(self, executor):
        """Successful - Error"""
        future = executor.submit(function, FunctionException, 1, None)

        with pytest.raises(ValueError):
            assert future.successful()

        future.wait()
        assert future.successful() is False

        with pytest.raises(FunctionException):
            future.get()
