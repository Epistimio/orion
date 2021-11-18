import dataclasses
import logging
import pickle
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Manager, Process
from multiprocessing.pool import AsyncResult, ExceptionWithTraceback, MaybeEncodingError
from multiprocessing.pool import Pool as PyPool
from multiprocessing.pool import RemoteTraceback, _helper_reraises_exception
from queue import Empty

import cloudpickle

from orion.executor.base import AsyncException, AsyncResult, BaseExecutor

log = logging.getLogger(__name__)


def _couldpickle_exec(payload):
    function, args, kwargs = pickle.loads(payload)
    result = function(*args, **kwargs)
    return cloudpickle.dumps(result)


class _Process(Process):
    """Process that cannot be a daemon"""

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


def _worker(
    inqueue,
    outqueue,
    initializer=None,
    initargs=(),
    maxtasks=None,
    wrap_exception=False,
):
    if (maxtasks is not None) and not (isinstance(maxtasks, int) and maxtasks >= 1):
        raise AssertionError("Maxtasks {!r} is not valid".format(maxtasks))
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, "_writer"):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        try:
            task = get()
        except (EOFError, OSError):
            log.debug("worker got EOFError or OSError -- exiting")
            break

        if task is None:
            log.debug("worker got sentinel -- exiting")
            break

        job, i, func, args, kwds = task
        try:
            result = (True, func(*args, **kwds))
        except KeyboardInterrupt as e:
            if wrap_exception and func is not _helper_reraises_exception:
                e = ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)
        except Exception as e:
            if wrap_exception and func is not _helper_reraises_exception:
                e = ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)
        try:
            put((job, i, result))
        except Exception as e:
            wrapped = MaybeEncodingError(e, result[1])
            log.debug("Possible encoding error while sending result: %s" % (wrapped))
            put((job, i, (False, wrapped)))

        task = job = result = func = args = kwds = None
        completed += 1
    log.debug("worker exiting after %d tasks" % completed)


class Pool(PyPool):
    """Custom pool that does not set its worker as daemon process"""

    @staticmethod
    def Process(*args, **kwds):
        import sys

        v = sys.version_info

        #  < 3.8 use self._ctx
        # >= 3.8 ctx as an argument
        if v.major == 3 and v.minor >= 8:
            args = args[1:]

        return _Process(*args, **kwds)

    @staticmethod
    def _repopulate_pool_static(
        ctx,
        Process,
        processes,
        pool,
        inqueue,
        outqueue,
        initializer,
        initargs,
        maxtasksperchild,
        wrap_exception,
    ):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(processes - len(pool)):
            w = Process(
                ctx,
                target=_worker,
                args=(
                    inqueue,
                    outqueue,
                    initializer,
                    initargs,
                    maxtasksperchild,
                    wrap_exception,
                ),
            )
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            pool.append(w)
            log.debug("added worker")


class _Future:
    """Wraps a python AsyncResult"""

    def __init__(self, future, cloudpickle=False):
        self.future = future
        self.cloudpickle = cloudpickle

    def get(self, timeout=None):
        r = self.future.get(timeout)
        return pickle.loads(r) if self.cloudpickle else r

    def wait(self, timeout=None):
        return self.future.wait(timeout)

    def ready(self):
        return self.future.ready()

    def succesful(self):
        return self.future.succesful()


class Multiprocess(BaseExecutor):
    """Simple multiprocess executor that wraps ``multiprocessing.Pool``."""

    def __init__(self, n_workers, **kwargs):
        super().__init__(n_workers, **kwargs)
        self.pool = Pool(n_workers)

    def __del__(self):
        self.pool.terminate()

    def __getstate__(self):
        state = super(Multiprocess, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(Multiprocess, self).__setstate__(state)

    def __enter__(self):
        self.pool.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)

    def submit(self, function, *args, **kwargs) -> AsyncResult:
        return self._submit_cloudpickle(function, *args, **kwargs)

    def _submit_python(self, function, *args, **kwargs) -> AsyncResult:
        return _Future(self.pool.apply_async(function, args=args, kwds=kwargs))

    def _submit_cloudpickle(self, function, *args, **kwargs) -> AsyncResult:
        payload = cloudpickle.dumps((function, args, kwargs))
        return _Future(self.pool.apply_async(_couldpickle_exec, (payload,)), True)

    def wait(self, futures):
        return [future.get() for future in futures]

    def _find_future_exception(self, future):
        for _, status in self.futures.items():
            if id(status.future) == id(future):
                return status.exception

    def async_get(self, futures, timeout=None):
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
