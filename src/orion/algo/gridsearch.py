"""
Grid Search
===========
"""
from __future__ import annotations

import itertools
import logging

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Categorical, Dimension, Fidelity, Integer, Real, Space
from orion.core.utils import format_trials

log = logging.getLogger(__name__)


def grid(dim: Dimension, num: int):
    """Build a one-dim grid of num points"""

    if dim.type == "categorical":
        # NOTE: Following lines have type errors, because we check the type using the `type`
        # attribute rather than using `isinstance`. This is the right thing to do, since the
        # TransformedDimension subclasses wouldn't be handled correctly using `isinstance`.
        # TODO: Would be nice for the different TransformedSpace subclasses to be generics w.r.t.
        # the type of the wrapped space. This way, we could annotate the functions below with e.g.
        # `Integer | TransformedSpace[Integer]`.
        return categorical_grid(dim, num)  # type: ignore
    elif dim.type == "integer":
        return discrete_grid(dim, num)  # type: ignore
    elif dim.type == "real":
        return real_grid(dim, num)  # type: ignore
    elif dim.type == "fidelity":
        return fidelity_grid(dim, num)  # type: ignore
    else:
        raise TypeError(
            "Grid Search only supports `real`, `integer`, `categorical` and `fidelity`: "
            f"`{dim.type}`\n"
            "For more information on dimension types, see "
            "https://orion.readthedocs.io/en/stable/user/searchspace.html"
        )


def fidelity_grid(dim: Fidelity, num: int):
    """Build fidelity grid, that is, only top value"""
    return [dim.interval()[1]]


def categorical_grid(dim: Categorical, num: int):
    """Build categorical grid, that is, all categories"""
    categories = dim.interval()
    if len(categories) != num:
        log.warning(
            f"Categorical dimension {dim.name} does not have {num} choices: {categories}. "
            "Will use {len(categories)} choices instead."
        )
    return categories


def discrete_grid(dim: Integer, num: int):
    """Build discretized real grid"""
    grid = real_grid(dim, num)

    _, b = dim.interval()

    discrete_grid = [int(numpy.round(grid[0]))]
    for v in grid[1:]:
        int_v = int(numpy.round(v))
        if int_v <= discrete_grid[-1]:
            int_v = discrete_grid[-1] + 1

        if int_v > b:
            log.warning(
                f"Cannot list {num} discrete values for {dim.name}. "
                "Will use {len(discrete_grid)} points instead."
            )
            break

        discrete_grid.append(int_v)

    return discrete_grid


def real_grid(dim: Real, num: int):
    """Build real grid"""
    if dim.prior_name.endswith("reciprocal"):
        a, b = dim.interval()
        return list(_log_grid(a, b, num))
    elif dim.prior_name.endswith("uniform"):
        a, b = dim.interval()
        return list(_lin_grid(a, b, num))
    else:
        raise TypeError(
            "Grid Search only supports `loguniform`, `uniform` and `choices`: "
            "`{}`".format(dim.prior_name)
        )


def _log_grid(a, b, num: int) -> numpy.ndarray:
    return numpy.exp(_lin_grid(numpy.log(a), numpy.log(b), num))


def _lin_grid(a, b, num: int) -> numpy.ndarray:
    return numpy.linspace(a, b, num=num)


class GridSearch(BaseAlgorithm):
    """Grid Search algorithm

    Parameters
    ----------
    n_values: int or dict
        Number of trials for each dimensions, or dictionary specifying number of trials for each
        dimension independently (name, n_values). For categorical dimensions, n_values will not be
        used, and all categories will be used to build the grid.
    """

    requires_type = None
    requires_dist = None
    requires_shape = "flattened"

    def __init__(
        self,
        space: Space,
        n_values: int | dict[str, int] = 100,
    ):
        super().__init__(space)
        self.n = 0
        self.n_values = n_values
        n_values_dict = (
            {name: n_values for name in self.space.keys()}
            if not isinstance(n_values, dict)
            else n_values
        )
        self.grid = self.build_grid(
            self.space, n_values_dict, getattr(self, "max_trials", 10000)
        )
        self.index = 0

    @staticmethod
    def build_grid(space: Space, n_values: dict[str, int], max_trials: int = 10000):
        """Build a grid of trials

        Parameters
        ----------
        n_values: int or dict
            Dictionary specifying number of trials for each dimension independently
            (name, n_values). For categorical dimensions, n_values will not be used, and all
            categories will be used to build the grid.
        max_trials: int
            Maximum number of trials for the grid. If n_values lead to more trials than max_trials,
            the n_values will be adjusted down. Will raise ValueError if it is impossible to build
            a grid smaller than max_trials (for instance if choices are too large).

        """
        adjust = 0
        n_trials = float("inf")
        coordinates: list[list] = []

        while n_trials > max_trials:
            coordinates = []
            capped_values = []
            for name, dim in space.items():
                capped_value = max(n_values[name] - adjust, 1)
                capped_values.append(capped_value)
                coordinates.append(list(grid(dim, capped_value)))

            if all(value <= 1 for value in capped_values):
                raise ValueError(
                    f"Cannot build a grid smaller than {max_trials}. "
                    "Try reducing the number of choices in categorical dimensions."
                )

            n_trials = numpy.prod([len(dim_values) for dim_values in coordinates])
            # TODO: Use binary search instead of incrementing by one.
            adjust += 1

        if adjust > 1:
            log.warning(
                f"`n_values` reduced by {adjust-1} to limit number of trials below {max_trials}."
            )

        return list(itertools.product(*coordinates))

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["grid"] = self.grid
        state_dict["index"] = self.index
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        Parameters
        ----------
        state_dict: dict
            Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.grid = state_dict["grid"]
        self.index = state_dict["index"]

    def suggest(self, num):
        """Return the entire grid of suggestions

        Returns
        -------
        list of trials or None
            A list of lists representing trials suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        """
        trials = []
        while len(trials) < num and self.index < len(self.grid):
            trial = format_trials.tuple_to_trial(self.grid[self.index], self.space)
            if not self.has_suggested(trial):
                self.register(trial)
                trials.append(trial)
            self.index += 1

        return trials

    @property
    def is_done(self):
        """Return True when all grid has been covered."""
        # NOTE: GridSearch doesn't care about the space cardinality, it can just check if the grid
        # has been completely explored.
        return (
            self.has_completed_max_trials
            or self.grid is not None
            and self.n_suggested >= len(self.grid)
        )

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        # NOTE: Override parent method to ignore `seed`
        return {self.__class__.__name__.lower(): {"n_values": self.n_values}}
