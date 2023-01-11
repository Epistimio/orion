"""
Base Search Algorithm
=====================

Formulation of a general search algorithm with respect to some objective.
Algorithm implementations must inherit from `orion.algo.base.BaseAlgorithm`.

Algorithms can be created using `algo_factory.create()`.

Examples
--------
>>> algo_factory.create('random', space, seed=1)
>>> algo_factory.create('some_fancy_algo', space, **some_fancy_algo_config)

"""
from __future__ import annotations

import inspect
import logging
from abc import abstractmethod
from typing import Any

from orion.algo.base.registry import Registry
from orion.algo.space import Space
from orion.core.utils import GenericFactory
from orion.core.utils.flatten import flatten
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class BaseAlgorithm:
    """Base class describing what an algorithm can do.

    Parameters
    ----------
    space : `orion.algo.space.Space`
       Definition of a problem's parameter space.
    kwargs : dict
       Tunable elements of a particular algorithm, a dictionary from
       hyperparameter names to values.

    Notes
    -----
    We are using the No Free Lunch theorem's [1]_[3]_ formulation of an
    `BaseAlgorithm`.
    We treat it as a part of a procedure which in each iteration suggests a
    sample of the parameter space of the problem as a candidate solution and observes
    the results of its evaluation.

    **Developer Note**: Each algorithm's complete specification, i.e.  implementation of its methods
    and parameters of its own, lies in a separate concrete algorithm class, which must be an
    **immediate** subclass of `BaseAlgorithm`. [The reason for this is current implementation of
    `orion.core.utils.Factory` metaclass which uses `BaseAlgorithm.__subclasses__()`.] Second, one
    must declare an algorithm's own parameters (tunable elements which could be set by
    configuration). This is done by passing them to `BaseAlgorithm.__init__()` by calling Python's
    super with a `Space` object as a positional argument plus algorithm's own parameters as keyword
    arguments. The keys of the keyword arguments passed to `BaseAlgorithm.__init__()` are
    interpreted as the algorithm's parameter names. So for example, a subclass could be as simple
    as this (regarding the logistics, not an actual algorithm's implementation):

    Examples
    --------
    .. code-block:: python
       :linenos:
       :emphasize-lines: 7

       from orion.algo.base import BaseAlgorithm
       from orion.algo.space import (Integer, Space)

       class MySimpleAlgo(BaseAlgorithm):

           def __init__(self, space, multiplier=1, another_param="a string param"):
               super().__init__(space, multiplier=multiplier, another_param=another_param)

           def suggest(self, num=1):
               print(self.another_param)
               return list(map(lambda x: tuple(map(lambda y: self.multiplier * y, x)),
                               self.space.sample(num)))

           def observe(self, points, results):
               pass

       dim = Integer('named_param', 'norm', 3, 2, shape=(2, 3))
       s = Space()
       s.register(dim)

       algo = MySimpleAlgo(s, 2, "I am just sampling!")
       algo.suggest()

    References
    ----------
    .. [1] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Optimization,”
       IEEE Transactions on Evolutionary Computation, vol. 1, no. 1, pp. 67–82, Apr. 1997.
    .. [2] W. G. Macready and D. H. Wolpert, “What Makes An Optimization Problem Hard?,”
       Complexity, vol. 1, no. 5, pp. 40–46, 1996.
    .. [3] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Search,”
       Technical Report SFI-TR-95-02-010, Santa Fe Institute, 1995.

    """

    deterministic = False
    requires_type = None
    requires_shape = None
    requires_dist = None

    max_trials: int | None = None

    def __init__(self, space: Space, **kwargs):
        log.debug(
            "Creating Algorithm object of %s type with parameters:\n%s",
            type(self).__name__,
            kwargs,
        )
        self._space = space
        if kwargs:
            param_names = list(kwargs)
        else:
            init_signature = inspect.signature(type(self).__init__)
            param_names = [
                name
                for name, param in init_signature.parameters.items()
                if name not in ["self", "space"]
                and param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]
            ]
        self._param_names = param_names
        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            setattr(self, varname, param)

        # TODO: move this inside an initialization function.
        if hasattr(self, "seed"):
            self.seed_rng(self.seed)

        self.registry = Registry()

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {"registry": self.registry.state_dict}

    def set_state(self, state_dict: dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.registry.set_state(state_dict["registry"])

    def get_id(
        self, trial: Trial, ignore_fidelity: bool = False, ignore_parent: bool = False
    ) -> str:
        """Return unique hash for a trials based on params

        The trial is assumed to be in the optimization space of the algorithm.

        Parameters
        ----------
        trial : Trial
            trial from a `orion.algo.space.Space`.
        ignore_fidelity: bool, optional
            If True, the fidelity dimension is ignored when computing a unique hash for
            the trial. Defaults to False.
        ignore_parent: bool, optional
            If True, the parent id is ignored when computing a unique hash for
            the trial. Defaults to False.

        """
        return trial.compute_trial_hash(
            trial,
            ignore_fidelity=ignore_fidelity,
            ignore_lie=True,
            ignore_parent=ignore_parent,
        )

    @property
    def fidelity_index(self) -> str | None:
        """Returns the name of the first fidelity dimension if there is one, otherwise `None`."""
        fidelity_dims = [dim for dim in self.space.values() if dim.type == "fidelity"]
        if fidelity_dims:
            return fidelity_dims[0].name
        return None

    @abstractmethod
    def suggest(self, num: int) -> list[Trial]:
        """Suggest a `num` of new sets of parameters.

        Parameters
        ----------
        num: int
            Number of points to suggest. The algorithm may return less than the number of points
            requested.

        Returns
        -------
        list of trials or None
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        IMPORTANT: Algorithms must call `self.register(trial)` for every trial that is returned by
        this method. This is important for the algorithm to be able to keep track of the trials it
        has suggested/observed, and for the auto-generated unit-tests to pass.
        """

    def observe(self, trials: list[Trial]) -> None:
        """Observe the `results` of the evaluation of the `trials` in the
        process defined in user's script.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            if not self.has_observed(trial):
                self.register(trial)

    def register(self, trial: Trial) -> None:
        """Save the trial as one suggested or observed by the algorithm.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           a Trial from `self.space`.
        """
        self.registry.register(trial)

    @property
    def n_suggested(self) -> int:
        """Number of trials suggested by the algorithm"""
        return len(self.registry)

    @property
    def n_observed(self) -> int:
        """Number of completed trials observed by the algorithm."""
        return sum(self.has_observed(trial) for trial in self.registry)

    def has_suggested(self, trial: Trial) -> bool:
        """Whether the algorithm has suggested a given point.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial from a `orion.algo.space.Space`.

        Returns
        -------
        bool
            True if the trial was suggested by the algo, False otherwise.

        """
        return self.registry.has_suggested(trial)

    def has_observed(self, trial: Trial) -> bool:
        """Whether the algorithm has observed a given point objective.

        This only counts observed completed trials.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial object to retrieve from the database

        Returns
        -------
        bool
            True if the trial's objective was observed by the algo, False otherwise.

        """
        return self.registry.has_observed(trial)

    @property
    def is_done(self) -> bool:
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        return self.has_completed_max_trials or self.has_suggested_all_possible_values()

    # pylint: disable=invalid-name
    def has_suggested_all_possible_values(self) -> bool:
        """Returns True if the algorithm has more trials in its registry than the number of possible
        values in the search space.

        If there is a fidelity dimension in the search space, only the trials with the maximum
        fidelity value are counted.
        """
        fidelity_index = self.fidelity_index
        if fidelity_index is not None:
            n_suggested_with_max_fidelity = 0
            fidelity_dim = self.space[fidelity_index]
            _, max_fidelity_value = fidelity_dim.interval()
            for trial in self.registry:
                fidelity_value = flatten(trial.params)[fidelity_index]
                if fidelity_value >= max_fidelity_value:
                    n_suggested_with_max_fidelity += 1
            return n_suggested_with_max_fidelity >= self.space.cardinality

        return self.n_suggested >= self.space.cardinality

    @property
    def has_completed_max_trials(self) -> bool:
        """Returns True if the algorithm has a `max_trials` attribute, and has completed more trials
        than its value.
        """
        if self.max_trials is None:
            return False

        fidelity_index = self.fidelity_index
        max_fidelity_value = None
        # When a fidelity dimension is present, we only count trials that have the maximum value.
        if fidelity_index is not None:
            _, max_fidelity_value = self.space[fidelity_index].interval()

        def _is_completed(trial: Trial) -> bool:
            if fidelity_index is None:
                return trial.status == "completed"
            return (
                trial.status == "completed"
                and flatten(trial.params)[fidelity_index] >= max_fidelity_value
            )

        return sum(map(_is_completed, self.registry)) >= self.max_trials

    # pylint: disable=unused-argument
    def score(self, trial: Trial) -> float:
        """Allow algorithm to evaluate `trial` based on a prediction about
        this parameter set's performance.

        By default, return the same score any parameter (no preference).

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial object to retrieve from the database

        Returns
        -------
        A subjective measure of expected performance.

        """
        return 0

    # pylint: disable=unused-argument
    def judge(self, trial: Trial, measurements: Any) -> dict | None:
        """Inform an algorithm about online `measurements` of a running trial.

        This method is to be used as a callback in a client-server communication
        between user's script and a orion's worker using a `BaseAlgorithm`.
        Data returned from this method must be serializable and will be used as
        a response to the running environment. Default response is None.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial object to retrieve from the database

        Notes
        -----

        Calling algorithm to `judge` a `point` based on its online `measurements` will effectively
        change a state in the algorithm (like a reinforcement learning agent's hidden state or an
        automatic early stopping mechanism's regression), which it may change the value of the
        property `should_suspend`.

        Returns
        -------
        None or a serializable dictionary containing named data

        """
        return None

    def should_suspend(self, trial: Trial) -> bool:
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return False

    @property
    def configuration(self) -> dict[str, Any]:
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        By default, returns a dictionary containing the attributes of `self` which are also
        constructor arguments.
        """
        dict_form = {}
        for attrname in self._param_names:
            if attrname.startswith("_"):  # Do not log _space or others in conf
                continue
            dict_form[attrname] = getattr(self, attrname)

        return {self.__class__.__name__.lower(): dict_form}

    @property
    def space(self) -> Space:
        """Domain of problem associated with this algorithm's instance."""
        return self._space

    @property
    def unwrapped(self):
        """Return the algorithm without transforms"""
        return self


algo_factory = GenericFactory(BaseAlgorithm)
