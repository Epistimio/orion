import traceback

from orion.executor.base import (
    AsyncException,
    AsyncResult,
    BaseExecutor,
    ExecutorClosed,
    Future,
)

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class _Future(Future):
    def __init__(self, future):
        self.future = future
        self.exception = None

    def get(self, timeout=None):
        if self.exception:
            raise self.exception
        try:
            return ray.get(self.future,timeout=timeout)
        except ray.exceptions.GetTimeoutError as e:
            print(e)

    def wait(self, timeout=None):
        try:
            ray.get(self.future, timeout = timeout)
        except ray.exceptions.GetTimeoutError:
            pass
        except Exception as e:
            self.exception = e

    def ready(self):
        return self.future.future().done()

    def successful(self):
        # Python 3.6 raise assertion error
        if not self.ready():
            raise ValueError()

        return self.future.successful()


class Ray(BaseExecutor):
    def __init__(self, n_workers=-1, **config):
        super().__init__(n_workers=n_workers)
        
        if not HAS_RAY:
            raise ImportError("Ray must be installed to use Ray executor.")
        self.config = config
        ray.init()
        print("Ray was initiated")
    
    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)

    def __del__(self):
        ray.shutdown()

    def __enter__(self):
        return self

    def submit(self, function, *args, **kwargs):
        try:
            remote_g = ray.remote(function)
            return _Future(remote_g.remote(*args, **kwargs))
        except Exception as e:
            if str(e).startswith(
                "Tried sending message after closing.  Status: closed"
            ):
                raise ExecutorClosed() from e

            raise
    def wait(self, futures):
        return [future.get() for future in futures]

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

    def __exit__(self, exc_type, exc_value, traceback):
        ray.shutdown()
        super().__exit__(exc_type, exc_value, traceback)
