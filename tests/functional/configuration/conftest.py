"""Common fixtures and utils for configuration tests."""
from orion.algo.base import BaseAlgorithm
from orion.algo.base.parallel_strategy import ParallelStrategy


def __init__(self, *args, **params):
    """Initialize the parent class"""
    self.params = params
    super(type(self), self).__init__(*args, **params)


def stub(self, *args, **kwargs):
    """Just a stub..."""


def configuration(self):
    """Configuration based on self.params"""
    return {self.__class__.__name__.lower(): self.params}


# Keep pointers to classes so that they are not garbage collected.
algo_classes = []
for char in "ABCDE":
    algo_class = type(f"A{char}", (BaseAlgorithm,), {"suggest": stub, "observe": stub})

    algo_classes.append(algo_class)


# Keep pointers to classes so that they are not garbage collected.
strategy_classes = []
for char in "ABCDE":
    strategy_class = type(
        f"S{char}", (ParallelStrategy,), {"observe": stub, "__init__": __init__}
    )

    strategy_class.configuration = property(configuration)

    strategy_classes.append(strategy_class)
