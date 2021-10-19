from orion.executor.base import BaseExecutor

try:
    from dask.distributed import Client, get_client, get_worker, rejoin, secede, TimeoutError

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


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
        return results

    def waitone(self, futures, timeout=0.01):
        if len(futures) == 0:
            return []

        finished = []
        tobe_removed = []

        def add_result(fut, result):
            finished.append(result)
            tobe_removed.append(fut)

        def check_withtimeout(fut):
            try:
                return fut.result(timeout)
            except TimeoutError:
                return None

        while not finished:
            # check with a timeout i.e wait for a bit & check results
            result = check_withtimeout(futures[0])
            if result:
                add_result(futures[0], result)

            for future in futures[1:]:
                # check without waiting (first future already waited)
                # the result is ready so timeout is ignored
                if future.done():
                    result = future.result(timeout)
                    add_result(future, result)

        for future in tobe_removed:
            futures.remove(future)

        return finished

    def submit(self, function, *args, **kwargs):
        return self.client.submit(function, *args, **kwargs, pure=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super(Dask, self).__exit__(exc_type, exc_value, traceback)
