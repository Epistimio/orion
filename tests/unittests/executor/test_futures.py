import time

import pytest

from orion.executor.dask_backend import Dask
from orion.executor.multiprocess_backend import PoolExecutor


def multiprocess(n):
    """Create a Pool using the multiprocess backend"""
    return PoolExecutor(n, "multiprocess")


def thread(n):
    """Create a Pool using the threading backend"""
    return PoolExecutor(n, "threading")


backends = [thread, multiprocess, Dask]


class FunctionException(Exception):
    """Special exception for testing
    so we are sure we are catching the right exception

    """

    pass


def function(exception, sleep_time, result):
    """Simple test function"""
    time.sleep(sleep_time)
    if exception:
        raise exception
    return result


@pytest.mark.parametrize("backend", backends)
class TestFutures:
    """Test Future interface to make sure all backend behave the same"""

    def test_future_get_ok(self, backend):
        """Get - OK"""
        with backend(1) as executor:
            future = executor.submit(function, None, 0, 1)
            assert future.get() == 1

    def test_future_get_exception(self, backend):
        """Get - Error"""
        with backend(1) as executor:
            future = executor.submit(function, FunctionException, 0, None)

            with pytest.raises(FunctionException):
                future.get()

    def test_future_get_timeout(self, backend):
        """Get - Timeout"""
        with backend(1) as executor:
            future = executor.submit(function, None, 1, 1)

            with pytest.raises(TimeoutError):
                future.get(0.01) == 1

    def test_future_wait_ok(self, backend):
        """Wait - OK"""
        with backend(1) as executor:
            future = executor.submit(function, None, 0.1, 1)

            assert future.ready() is False
            future.wait()
            assert future.ready() is True

    def test_future_wait_error(self, backend):
        """Wait - Error"""
        with backend(1) as executor:
            future = executor.submit(function, FunctionException, 0.1, None)

            assert future.ready() is False
            future.wait()
            assert future.ready() is True

            with pytest.raises(FunctionException):
                future.get()

    def test_future_wait_timeout(self, backend):
        """Wait - Timeout"""
        with backend(1) as executor:
            future = executor.submit(function, FunctionException, 1, None)

            assert future.ready() is False
            future.wait(0.01)
            assert future.ready() is False
            future.wait()
            assert future.ready() is True

    def test_future_ready_ok(self, backend):
        """Ready - OK"""
        with backend(1) as executor:
            future = executor.submit(function, None, 1, 1)
            assert future.ready() is False
            future.wait()
            assert future.ready() is True

    def test_future_ready_error(self, backend):
        """Ready - Error"""
        with backend(1) as executor:
            future = executor.submit(function, FunctionException, 0.1, None)
            assert future.ready() is False
            future.wait()
            assert future.ready() is True

            with pytest.raises(FunctionException):
                future.get()

    def test_future_successful_ok(self, backend):
        """Successful - OK"""
        with backend(1) as executor:
            future = executor.submit(function, None, 1, 1)

            with pytest.raises(ValueError):
                assert future.successful()

            future.wait()
            assert future.successful() is True

    def test_future_sucessful_error(self, backend):
        """Successful - Error"""
        with backend(1) as executor:
            future = executor.submit(function, FunctionException, 1, None)

            with pytest.raises(ValueError):
                assert future.successful()

            future.wait()
            assert future.successful() is False

            with pytest.raises(FunctionException):
                future.get()
