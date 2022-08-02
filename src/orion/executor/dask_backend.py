import traceback

from orion.executor.base import (
    AsyncException,
    AsyncResult,
    BaseExecutor,
    ExecutorClosed,
    Future,
)

try:
    import dask.distributed
    from dask.distributed import Client, get_client, get_worker, rejoin, secede

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class _Future(Future):
    """Wraps a Dask Future"""

    def __init__(self, future):
        self.future = future
        self.exception = None

    def get(self, timeout=None):
        if self.exception:
            raise self.exception

        try:
            return self.future.result(timeout)
        except dask.distributed.TimeoutError as e:
            raise TimeoutError() from e

    def wait(self, timeout=None):
        try:
            self.future.result(timeout)
        except dask.distributed.TimeoutError:
            pass
        except Exception as e:
            self.exception = e

    def ready(self):
        return self.future.done()

    def successful(self):
        if not self.future.done():
            raise ValueError()

        return self.future.exception() is None


class Dask(BaseExecutor):
    """Wrapper around the dask client.

    .. warning::

       The Dask executor can be pickled and used inside a subprocess,
       the pickled client will use the main client that was spawned in the main process,
       but you cannot spawn clients inside a subprocess.

    """

    def __init__(self, n_workers=-1, client=None, **config):
        super().__init__(n_workers=n_workers)

        if not HAS_DASK:
            raise ImportError("Dask must be installed to use Dask executor.")

        self.config = config
        if client is None:
            client = Client(**self.config)

        self.client = client

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
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

        for i, future in enumerate(futures):
            if timeout and i == 0:
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
        try:
            return _Future(self.client.submit(function, *args, **kwargs, pure=False))
        except Exception as e:
            if str(e).startswith(
                "Tried sending message after closing.  Status: closed"
            ):
                raise ExecutorClosed() from e

            raise

    def __del__(self):
        # This is necessary because if the factory constructor fails
        # __del__ is executed right away but client might not be set
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)
