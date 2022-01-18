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


def function(exception, sleep_time, result):
    """Simple test function"""
    time.sleep(sleep_time)
    if exception:
        raise exception
    return result


@pytest.mark.parametrize("backend", backends)
def test_future(backend):
    """Make sure the future interface is harmonized"""
    with backend(5) as executor:

        # Get - OK
        future = executor.submit(function, None, 0, 1)
        assert future.get() == 1

        # Get - Error
        future = executor.submit(function, RuntimeError, 0, None)

        with pytest.raises(RuntimeError):
            future.get()

        # Get - Timeout
        future = executor.submit(function, None, 1, 1)

        with pytest.raises(TimeoutError):
            future.get(0.01) == 1

        # Wait - OK
        future = executor.submit(function, None, 0.1, 1)

        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        # Wait - Error
        future = executor.submit(function, RuntimeError, 0.1, None)

        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        with pytest.raises(RuntimeError):
            future.get()

        # Wait - Timeout
        future = executor.submit(function, RuntimeError, 1, None)

        assert future.ready() is False
        future.wait(0.01)
        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        # Ready - OK
        future = executor.submit(function, None, 1, 1)
        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        # Ready - Error
        future = executor.submit(function, RuntimeError, 0.1, None)
        assert future.ready() is False
        future.wait()
        assert future.ready() is True

        with pytest.raises(RuntimeError):
            future.get()

        # Successful - OK
        future = executor.submit(function, None, 1, 1)

        with pytest.raises(ValueError):
            assert future.successful()

        future.wait()
        assert future.successful() is True

        # Successful - Error
        future = executor.submit(function, RuntimeError, 1, None)

        with pytest.raises(ValueError):
            assert future.successful()

        future.wait()
        assert future.successful() is False

        with pytest.raises(RuntimeError):
            future.get()
