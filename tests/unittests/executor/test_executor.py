import pytest

from orion.executor.multiprocess_backend import Multiprocess
from orion.executor.dask_backend import Dask
from orion.executor.single_backend import SingleExecutor


backends = [Multiprocess, Dask, SingleExecutor]

def function(a, b, c):
    return a + b * c


class BadException(Exception):
    pass


def bad_function(a, b, c):
    raise BadException()


@pytest.mark.parametrize('backend', backends)
def test_execute_function(backend):
    with backend(5) as executor:
        future = executor.submit(function, 1, 2, c=3)
        assert executor.wait([future]) == [7]


@pytest.mark.parametrize('backend', backends)
def test_execute_bad_function(backend):
    with backend(5) as executor:
        future = executor.submit(bad_function, 1, 2, 3)
        with pytest.raises(BadException):
            executor.wait([future])


@pytest.mark.parametrize('backend', backends)
def test_execute_async(backend):
    with backend(5) as executor:
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]

        total_task = len(futures)
        results = executor.async_get(futures)

        assert len(results) < total_task, 'Not all tasks were completed'
        assert len(results) + len(futures) == total_task, 'Future were removed'


@pytest.mark.parametrize('backend', backends)
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

    all_results_async.sort()
    assert all_results_async == all_results


def nested_jobs(executor):
    with executor:
        print('nested_jobs sub')
        futures = [executor.submit(function, 1, 2, i) for i in range(10)]
        print('nested_jobs wait')
        all_results = executor.wait(futures)
    return sum(all_results)


@pytest.mark.parametrize('backend', [Dask, SingleExecutor])
def test_executor_is_serializable(backend):
    with backend(5) as executor:
        print('serialize sub')
        futures = [executor.submit(nested_jobs, executor) for _ in range(10)]
        print('serialize wait')
        all_results = executor.wait(futures)

    assert sum(all_results) == 1000
