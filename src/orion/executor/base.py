from orion.core.utils import Factory


class BaseExecutor:
    def __init__(self, n_workers, **kwargs):
        self.n_workers = n_workers

    def __getstate__(self):
        return dict(n_workers=self.n_workers)

    def __setstate__(self, state):
        self.n_workers = state["n_workers"]

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
