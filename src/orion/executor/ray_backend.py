import logging
import traceback

from orion.core.utils.module_import import ImportOptional
from orion.executor.base import (
    AsyncException,
    AsyncResult,
    BaseExecutor,
    ExecutorClosed,
    Future,
)

with ImportOptional("ray") as import_optional:
    import ray

HAS_RAY = not import_optional.failed

logger = logging.getLogger(__name__)


class _Future(Future):
    def __init__(self, future):
        self.future = future
        self.exception = None

    def get(self, timeout=None):
        if self.exception:
            raise self.exception
        try:
            return ray.get(self.future, timeout=timeout)
        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError() from e

    def wait(self, timeout=None):
        try:
            ray.get(self.future, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            pass
        except Exception as e:
            self.exception = e

    def ready(self):
        obj_ready = ray.wait([self.future])
        return len(obj_ready[0]) == 1

    def successful(self):
        # Python 3.6 raise assertion error
        if not self.ready():
            raise ValueError()

        return self.future.successful()


class Ray(BaseExecutor):
    def __init__(
        self,
        n_workers=-1,
        **config,
    ):
        super().__init__(n_workers=n_workers)
        self.initialized = False
        if not HAS_RAY:
            raise ImportError("Ray must be installed to use Ray executor.")
        self.config = config

        if not ray.is_initialized():
            ray.init(**self.config)
            self.initialized = True
            logger.debug("Ray was initiated with runtime_env : %s", **config)

    def close(self):
        if self.initialized:
            self.initialized = False
            ray.shutdown()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def submit(self, function, *args, **kwargs):
        if not ray.is_initialized():
            raise ExecutorClosed()

        remote_g = ray.remote(function)
        return _Future(remote_g.remote(*args, **kwargs))

    def wait(self, futures):
        return [future.get() for future in futures]

    def async_get(self, futures, timeout=None):
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
        self.close()
        super().__exit__(exc_type, exc_value, traceback)
