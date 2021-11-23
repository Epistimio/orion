import traceback
from multiprocessing import TimeoutError as PyTimeoutError
from multiprocessing import Value

from orion.executor.base import AsyncException, AsyncResult, BaseExecutor

try:
    from dask.distributed import (
        Client,
        TimeoutError,
        get_client,
        get_worker,
        rejoin,
        secede,
    )

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class _Future:
    """Wraps a Dask Future"""

    def __init__(self, future):
        self.future = future

    def get(self, timeout=None):
        try:
            return self.future.result(timeout)
        except TimeoutError as e:
            raise PyTimeoutError from e

    def wait(self, timeout=None):
        try:
            self.future.result(timeout)
        except TimeoutError:
            pass

    def ready(self):
        return self.future.done()

    def successful(self):
        if not self.future.done():
            raise ValueError()

        return self.future.exception() is None


class Dask(BaseExecutor):
    def __init__(self, n_workers=-1, client=None, **config):
        super(Dask, self).__init__(n_workers=n_workers)

        if not HAS_DASK:
            raise ImportError("Dask must be installed to use Dask executor.")

        self.config = config
        if client is None:
            client = Client(**self.config)

        self.client = client

    def __getstate__(self):
        return super(Dask, self).__getstate__()

    def __setstate__(self, state):
        super(Dask, self).__setstate__(state)
        self.client = get_client()

    @property
    def in_worker(self):
        try:
            get_worker()
            return True
        except ValueError:
            return False

    def wait(self, futures):
        if self.in_worker:
            secede()
        results = self.client.gather(list(futures))
        if self.in_worker:
            rejoin()
        return [r.get() for r in results]

    def async_get(self, futures, timeout=0.01):
        results = []
        tobe_deleted = []

        for future in futures:
            if timeout:
                future.wait(timeout)

            if future.ready():
                try:
                    results.append(AsyncResult(future, future.get()))
                except Exception as err:
                    results.append(AsyncException(future, err, traceback.format_exc()))

                tobe_deleted.append(future)

        for future in tobe_deleted:
            futures.remove(future)

        return results

    def submit(self, function, *args, **kwargs):
        return _Future(self.client.submit(function, *args, **kwargs, pure=False))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super(Dask, self).__exit__(exc_type, exc_value, traceback)
