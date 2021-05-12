from orion.executor.base import BaseExecutor

try:
    from dask.distributed import (
        Client,
        get_client,
        get_worker,
        rejoin,
        secede,
    )

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class Dask(BaseExecutor):
    def __init__(self, experiment=None, n_jobs=-1, client=None, **config):
        self.experiment = experiment

        if not HAS_DASK:
            raise ImportError("Dask must be installed to use Dask executor.")

        self.config = config
        if client is None:
            client = Client(**self.config)

        self.client = client

    def __getstate__(self):
        return dict()

    def __setstate__(self, state):
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
        return results

    def submit(self, function, *args, **kwargs):
        return self.client.submit(function, *args, **kwargs, pure=False)

    def __enter__(self):
        super(Dask, self).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super(Dask, self).__exit__(exc_type, exc_value, traceback)
