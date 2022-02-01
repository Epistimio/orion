# -*- coding: utf-8 -*-
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
import copy
import hashlib
import logging
from abc import ABCMeta, abstractmethod

from orion.algo.space import Fidelity
from orion.core.utils import GenericFactory, format_trials

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
    arguments. The keys of the keyword arguments passed to `BaseAlgorithm.__init__()` are interpreted
    as the algorithm's parameter names. So for example, a subclass could be as simple as this
    (regarding the logistics, not an actual algorithm's implementation):

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

    requires_type = None
    requires_shape = None
    requires_dist = None

    def __init__(self, space, **kwargs):
        log.debug(
            "Creating Algorithm object of %s type with parameters:\n%s",
            type(self).__name__,
            kwargs,
        )
        self._trials_info = {}  # Stores Unique Trial -> Result
        self._space = space
        self._param_names = list(kwargs.keys())
        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            setattr(self, varname, param)

        # TODO: move this inside an initialization function.
        if hasattr(self, "seed"):
            self.seed_rng(self.seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        pass

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {"_trials_info": copy.deepcopy(dict(self._trials_info))}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self._trials_info = state_dict.get("_trials_info")

    def format_trial(self, trial):
        """Format trial based on space transformations

        This will apply the reverse transformation on the trial and then
        transform it again.

        Some transformations are lossy and thus the trials suggested by the algorithm could
        be different when returned to `observe`. Using `format_trial` makes it possible
        for the algorithm to see the final version of the trial after back and forth
        transformations. This way it can recognise the trial in `observe` and also
        avoid duplicates that would have gone unnoticed during suggestion.

        Parameters
        ----------
        trial : `orion.core.worker.trial.Trial`
            Trial from a `orion.algo.space.Space`.
        """

        if hasattr(self.space, "transform"):
            trial = self.space.transform(self.space.reverse(trial))

        return trial

    def get_id(self, trial, ignore_fidelity=False, ignore_parent=False):
        """Return unique hash for a trials based on params

        The trial is assumed to be in the transformed space if the algorithm is working in a
        transformed space.

        Parameters
        ----------
        point : tuples of array-likes
            Points from a `orion.algo.space.Space`.
        ignore_fidelity: bool, optional
            If True, the fidelity dimension is ignored when computing a unique hash for
            the trial. Defaults to False.
        ignore_parent: bool, optional
            If True, the parent id is ignored when computing a unique hash for
            the trial. Defaults to False.

        """

        # Apply transforms and reverse to see data as it would come from DB
        # (Some transformations looses some info. ex: Precision transformation)

        # Compute trial hash in the client-facing format.
        if hasattr(self.space, "reverse"):
            trial = self.space.reverse(trial)

        return trial.compute_trial_hash(
            trial,
            ignore_fidelity=ignore_fidelity,
            ignore_experiment=True,
            ignore_lie=True,
            ignore_parent=ignore_parent,
        )

    @property
    def fidelity_index(self):
        """Compute the dimension name of the space where fidelity is.

        Returns None if there is no fidelity dimension.
        """

        def _is_fidelity(dim):
            return dim.type == "fidelity"

        fidelity_dim = [dim for dim in self.space.values() if _is_fidelity(dim)]
        if fidelity_dim:
            return fidelity_dim[0].name

        return None

    @abstractmethod
    def suggest(self, num):
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

        """
        pass

    def observe(self, trials):
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

    def register(self, trial):
        """Save the trial as one suggested or observed by the algorithm.

        The trial objectives may change without the algorithm having actually observed it.
        In order to detect this, we assign a tuple ``(trial and trial.objective)``
        to the key ``self.get_id(trial)`` so that if the objective was not observed, we
        will see that second item of the tuple is ``None``.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial from a `orion.algo.space.Space`.

        """
        self._trials_info[self.get_id(trial)] = (
            copy.deepcopy(trial),
            format_trials.get_trial_results(trial) if trial.objective else None,
        )

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm"""
        return len(self._trials_info)

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm"""
        return sum(bool(point[1] is not None) for point in self._trials_info.values())

    def has_suggested(self, trial):
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
        return self.get_id(trial) in self._trials_info

    def has_observed(self, trial):
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
        return (
            self.get_id(trial) in self._trials_info
            and self._trials_info[self.get_id(trial)][1] is not None
        )

    @property
    def is_done(self):
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if self.n_suggested >= self.space.cardinality:
            return True

        if self.n_observed >= getattr(self, "max_trials", float("inf")):
            return True

        return False

    def score(self, trial):  # pylint:disable=no-self-use,unused-argument
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance.

        By default, return the same score any parameter (no preference).

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
           Trial object to retrieve from the database

        Returns
        -------
        A subjective measure of expected perfomance.

        """
        return 0

    def judge(self, trial, measurements):  # pylint:disable=no-self-use,unused-argument
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

    def should_suspend(self, trial):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return False

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith("_"):  # Do not log _space or others in conf
                continue
            dict_form[attrname] = getattr(self, attrname)

        return {self.__class__.__name__.lower(): dict_form}

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance."""
        return self._space

    @space.setter
    def space(self, space):
        """Set space."""
        self._space = space


algo_factory = GenericFactory(BaseAlgorithm)
