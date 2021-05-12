from orion.core.utils import Factory


class BaseExecutor:
    def __init__(self, n_jobs, **kwargs):
        self.n_jobs = n_jobs

    def __getstate__(self):
        return dict(n_jobs=self.n_jobs)

    def __setstate__(self, state):
        self.n_jobs = state["n_jobs"]

    def wait(self, futures):
        pass

    def submit(self, function, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# pylint: disable=too-few-public-methods,abstract-method
class Executor(BaseExecutor, metaclass=Factory):
    """TODO"""
