""" ABC for a wrapper around an Algorithm. """
from abc import ABC
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Union, Any, TypeVar, Generic
import numpy as np

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.worker.trial import Trial


# Type aliases
Point = Union[Tuple, List[float], np.ndarray]
Results = List[Dict]
AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)

class AlgoWrapper(BaseAlgorithm, Generic[AlgoType], ABC):
    """Base class for a Wrapper around an algorithm.
    """
    def __init__(self, space: Space, **kwargs):
        # NOTE: This field is created automatically in the BaseAlgorithm class.
        self._algorithm: AlgoType
        super().__init__(space, **kwargs)

    @property
    def algorithm(self) -> AlgoType:
        """Returns the wrapped algorithm.

        Returns
        -------
        AlgoType
            The wrapped algorithm.
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: AlgoType) -> None:
        """Sets the wrapped algorithm. Can only be called once.

        Parameters
        ----------
        value : AlgoType
            The algorithm wrapped by this Wrapped.

        Raises
        ------
        RuntimeError
            If `self._algorithm` is already set.
        """
        _algorithm = getattr(self, "_algorithm", None)
        if _algorithm is not None and value is not _algorithm:
            raise RuntimeError("Can't change the value of `algorithm` after it's been set!")
        self._algorithm = value

    @property
    def unwrapped(self) -> BaseAlgorithm:
        """Returns the unwrapped algorithm (the root).

        Returns
        -------
        BaseAlgorithm
            The unwrapped `BaseAlgorithm` instance.
        """
        return self.algorithm.unwrapped
    
    def seed_rng(self, seed: Optional[int]) -> None:
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self) -> Dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        return self.algorithm.state_dict

    def set_state(self, state_dict: Dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.algorithm.set_state(state_dict)

    def suggest(self, num: int = 1) -> List[Point]:
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        return self.algorithm.suggest(num)

    def observe(self, points: List[Union[List, Tuple, np.ndarray]], results: List[Dict]):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        self.algorithm.observe(points, results)

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm"""
        return self.algorithm.n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm"""
        return self.algorithm.n_observed

    @property
    def is_done(self) -> bool:
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.algorithm.is_done

    def score(self, point: Point) -> float:
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        return self.algorithm.score(point)

    def judge(self, point: Point, measurements: Any) -> Optional[Dict]:
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        return self.algorithm.judge(point, measurements)

    @property
    def should_suspend(self) -> bool:
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return self.algorithm.should_suspend

    @property
    def configuration(self) -> Dict:
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.algorithm.configuration

    @contextmanager
    def warm_start_mode(self):
        """ Context manager that is used while using points from similar experiments to
        bootstrap (warm-start) the algorithm.

        The idea behing this is that we don't want the algorithm to modify its state the
        same way it would if it were observing regular points. For example, the number
        of "used" trials shouldn't increase, etc.

        New Algorithms or Algo wrappers can implement this method to control how the
        state of the algo is affected by observing trials from other tasks than the
        current (target) task.
        """
        with self.algorithm.warm_start_mode():
            yield

    @property
    def trials_info(self) -> Dict[str, Tuple[Point, Results]]:
        """ "read-only" property for the dict of points/trials of the wrapped algorithm.
        """
        return self._trials_info

    @property
    def _trials_info(self) -> Dict[str, Tuple[Point, Results]]:
        """ Proxy for the `_trials_info` field of the wrapped algorithm.

        NOTE: Adding this property just to make sure that if an algo wrapper doesn't
        override a BaseAlgorithm method (for example when new methods / properties get
        added to BaseAlgorithm but aren't yet overridden in AlgoWrapper), then things
        don't accidentally get changed in the Wrapper's `_trials_info` rather than in
        the wrapped algorithm's.
        """
        return self.algorithm._trials_info  # type: ignore

    @_trials_info.setter
    def _trials_info(self, value: Dict[str, Trial]) -> None:
        if self.algorithm is None:
            # this is being set as part of '__init__' of the base class: ignore.
            # This __trials_info_set attribute should be False.
            assert not getattr(self, "__trials_info_set", False)
            self.__trials_info_set = True
            return
        self.unwrapped._trials_info = value
