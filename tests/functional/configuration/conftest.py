"""Common fixtures and utils for configuration tests."""
from orion.algo.base import BaseAlgorithm, OptimizationAlgorithm
from orion.core.worker.strategy import BaseParallelStrategy, Strategy


def __init__(self, *args, **params):
    """Initialize the parent class"""
    self.params = params
    super(type(self), self).__init__(*args, **params)


def stub(self, *args, **kwargs):
    """Just a stub..."""
    pass


def configuration(self):
    """Configuration based on self.params"""
    return {self.__class__.__name__.lower(): self.params}


for char in "ABCDE":
    algo_class = type(f"A{char}", (BaseAlgorithm,), {"suggest": stub, "observe": stub})

    # Hack it into being discoverable
    OptimizationAlgorithm.types[algo_class.__name__.lower()] = algo_class


for char in "ABCDE":
    strategy_class = type(
        f"S{char}", (BaseParallelStrategy,), {"observe": stub, "__init__": __init__}
    )

    strategy_class.configuration = property(configuration)

    # Hack it into being discoverable
    Strategy.types[strategy_class.__name__.lower()] = strategy_class
