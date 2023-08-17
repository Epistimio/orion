import multiprocessing
import time

import pytest

from orion.executor.base import AsyncException, ExecutorClosed, executor_factory
from orion.executor.dask_backend import HAS_DASK, Dask
from orion.executor.multiprocess_backend import PoolExecutor
from orion.executor.single_backend import SingleExecutor


def multiprocess(n):
    return PoolExecutor(n, "multiprocess")


def thread(n):
    return PoolExecutor(n, "threading")


def skip_dask_if_not_installed(
    value, reason="Dask dependency is required for these tests."
):
    return pytest.param(
        value,
        marks=pytest.mark.skipif(
            not HAS_DASK,
            reason=reason,
        ),
    )


def xfail_dask_if_not_installed(
    value, reason="Dask dependency is required for these tests."
):
    return pytest.param(
        value,
        marks=pytest.mark.xfail(
            condition=not HAS_DASK, reason=reason, raises=ImportError
        ),
    )


executors = [
    "joblib",
    "poolexecutor",
    "singleexecutor",
    skip_dask_if_not_installed("dask"),
]

backends = [
    thread,
    multiprocess,
    SingleExecutor,
    skip_dask_if_not_installed(Dask),
]


def function(a, b, c):
    return a + b * c


def slow_function(a, b, c):
    time.sleep(5)
    return function(a, b, c)


class BadException(Exception):
    pass


def bad_function(a, b, c):
    raise BadException()


@pytest.mark.parametrize("backend", backends)
def test_execute_function(backend):
    with backend(5) as executor:
        future = executor.submit(function, 1, 2, c=3)
        assert executor.wait([future]) == [7]

    # Executor was closed at exit
    with pytest.raises(ExecutorClosed):
        executor.submit(function, 1, 2, c=3)


@pytest.mark.parametrize("backend", backends)
def test_execute_delete(backend):
    executor = backend(5)

    future = executor.submit(function, 1, 2, c=3)
    assert executor.wait([future]) == [7]

    executor.__del__()

    # Executor was closed when deleted
    with pytest.raises(ExecutorClosed):
        executor.submit(function, 1, 2, c=3)


@pytest.mark.parametrize("backend", backends)
def test_execute_bad_function(backend):
    with backend(5) as executor:
        future = executor.submit(bad_function, 1, 2, 3)
        with pytest.raises(BadException):
            executor.wait([future])


@pytest.mark.parametrize("backend", backends)
def test_execute_async_exception(backend):
    with backend(5) as executor:
        futures = [executor.submit(bad_function, 1, 2, i) for i in range(20)]
        results = []

        # waiting should not raise exception
        while len(results) != 20:
            partial = executor.async_get(futures)
            results.extend(partial)

        # exception is raised when we try to fetch the result
        for result in results:
            with pytest.raises(BadException):
                _ = result.value


@pytest.mark.parametrize("backend", backends)
def test_execute_async(backend):
    with backend(5) as executor:
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]

        total_task = len(futures)
        results = executor.async_get(futures, timeout=1)

        assert len(results) > 0, "We got some results"
        assert len(futures) == total_task - len(results), "Finished futures got removed"


@pytest.mark.parametrize("backend", backends)
def test_execute_async_all(backend):
    """Makes sure wait can be reinplemented as a async_get"""
    all_results = []

    with backend(5) as executor:
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]
        all_results = executor.wait(futures)
    all_results.sort()

    # Async version
    all_results_async = []
    with backend(5) as executor:
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]

        results = True
        while results:
            results = executor.async_get(futures, timeout=1)
            all_results_async.extend(results)

    all_results_async = [a.value for a in all_results_async]
    all_results_async.sort()
    assert all_results_async == all_results


@pytest.mark.parametrize(
    "backend",
    [thread, multiprocess, xfail_dask_if_not_installed(Dask)],
)
def test_execute_async_timeout(backend):
    """Makes sure async_get does not wait after timeout"""
    with backend(5) as executor:
        futures = [executor.submit(slow_function, 1, 2, i) for i in range(10)]
        results = executor.async_get(futures, timeout=1)

        assert len(results) == 0, "No tasks had time to finish yet"
        assert len(futures) == 10, "All futures are still there"


@pytest.mark.parametrize("backend", backends)
def test_execute_async_bad(backend):
    """Makes sure async_get does not throw exceptions"""
    with backend(5) as executor:
        futures = [executor.submit(bad_function, 1, 2, i) for i in range(10)]

        results = []
        while futures:
            results.extend(executor.async_get(futures))

    for result in results:
        assert isinstance(result, AsyncException)

        with pytest.raises(BadException):
            result.value


def nested_jobs(executor):
    with executor:
        print("nested_jobs sub")
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]
        print("nested_jobs wait")
        all_results = executor.wait(futures)
    return sum(all_results)


@pytest.mark.parametrize("backend", [SingleExecutor])
def test_executor_is_serializable(backend):
    with backend(5) as executor:
        futures = [executor.submit(nested_jobs, executor) for _ in range(10)]
        all_results = executor.wait(futures)

    assert sum(all_results) == 1000


def proxy(*args):
    import subprocess

    subprocess.run(["echo", ""])


@pytest.mark.parametrize("backend", backends)
def test_multisubprocess(backend):
    with backend(5) as executor:
        futures = [executor.submit(proxy) for i in range(5)]

        results = executor.async_get(futures, timeout=2)

        for r in results:
            # access the results to make sure no exception is being
            # suppressed
            r.value


def nested(executor):
    futures = []

    for i in range(5):
        futures.append(executor.submit(function, 1, 2, 3))

    return sum(f.get() for f in futures)


@pytest.mark.parametrize("backend", backends)
def test_nested_submit(backend):
    with backend(5) as executor:
        futures = [executor.submit(nested, executor) for i in range(5)]

        results = executor.async_get(futures, timeout=2)

        for r in results:
            assert r.value == 35


def inc(a):
    return a + 1


def nested_pool():
    import multiprocessing.process as proc

    assert not proc._current_process._config.get("daemon")

    data = [1, 2, 3, 4, 5, 6]
    with multiprocessing.Pool(5) as p:
        result = p.map_async(inc, data)
        result.wait()
        data = result.get()

    return sum(data)


@pytest.mark.parametrize("backend", backends)
def test_nested_submit_pool(backend):
    if backend is Dask:
        pytest.xfail("Dask does not support nesting")

    with backend(5) as executor:
        futures = [executor.submit(nested_pool) for i in range(5)]

        results = executor.async_get(futures, timeout=2)

        for r in results:
            assert r.value == 27


@pytest.mark.parametrize("executor", executors)
def test_executors_have_default_args(executor):

    with executor_factory.create(executor):
        pass


@pytest.mark.parametrize("backend", backends)
def test_executors_del_does_not_raise(backend):
    # if executor init fails you can get very weird error messages,
    # because of the deleter trying to close unallocated resources.

    executor = backend(1)
    if hasattr(executor, "pool"):
        executor.pool.shutdown()
        del executor.pool
    elif hasattr(executor, "client"):
        executor.client.close()
        del executor.client

    del executor


def pytorch_workon(pid):
    import multiprocessing.process as proc

    assert not proc._current_process._config.get("daemon")

    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.FakeData(128, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=64)

    for i, _ in enumerate(loader):
        pass

    return i


def has_pytorch():
    try:
        import torch

        return True
    except:
        return False


@pytest.mark.parametrize("backend", backends)
def test_pytorch_dataloader(backend):
    if backend is Dask:
        pytest.xfail("Dask does not support nesting")

    if not has_pytorch():
        pytest.skip("Pytorch is not installed skipping")
        return

    with backend(2) as executor:
        futures = [executor.submit(pytorch_workon, i) for i in range(2)]

        results = executor.async_get(futures, timeout=2)

        for r in results:
            assert r.value == 1
