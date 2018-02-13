# -*- coding: utf-8 -*-
"""
:mod:`metaopt.algo.base` -- What is a search algorithm, optimizer of a process
==============================================================================

.. module:: base
   :platform: Unix
   :synopsis: Formulation of a general search algorithm with respect to some
      objective.

"""
from abc import (ABCMeta, abstractmethod)
import logging

from metaopt.core.utils import Factory

log = logging.getLogger(__name__)


class BaseAlgorithm(object, metaclass=ABCMeta):
    """Base class describing what an algorithm can do.

    Notes
    -----
    We are using the No Free Lunch theorem's [1]_[3]_ formulation of an
    `BaseAlgorithm`.
    We treat it as a part of a procedure which in each iteration suggests a
    sample of the parameter space of the problem as a candidate solution and observes
    the results of its evaluation.

    **Developer Note**: Each algorithm's complete specification, i.e.
    implementation of its methods and hyperparameters of its own, lies in a
    concrete algorithm class. Decorator (TODO) `declare_param` is provided in
    `metaopt.algo.base` which enables developer to declare hyperparameters
    of an algorithm implementation or wrapper.

    References
    ----------
    .. [1] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Optimization,”
       IEEE Transactions on Evolutionary Computation, vol. 1, no. 1, pp. 67–82, Apr. 1997.
    .. [2] W. G. Macready and D. H. Wolpert, “What Makes An Optimization Problem Hard?,”
       Complexity, vol. 1, no. 5, pp. 40–46, 1996.
    .. [3] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Search,”
       Technical Report SFI-TR-95-02-010, Santa Fe Institute, 1995.

    """

    def __init__(self, space, **hypers):
        """Declare problem's parameter space and set up algo's hyperparameters.

        Parameters
        ----------
        space : `metaopt.algo.space.Space`
           Definition of a problem's parameter space.
        hypers : dict
           Tunable elements of a particular algorithm, a dictionary from
           hyperparameter names to values.

        """
        log.debug("Creating Algorithm object of %s type with parameters:\n%s",
                  type(self).__name__, hypers)
        self._space = space
        self._hyper_names = list(hypers.keys())
        # Instantiate tunable parameters of an algorithm
        for varname, hyper in hypers.items():
            # Check if tunable element is another algorithm
            if isinstance(hyper, dict) and len(hyper) == 1:
                subalgo_type = list(hyper)[0]
                subalgo_hypers = hyper[subalgo_type]
                if isinstance(subalgo_hypers, dict):
                    hyper = OptimizationAlgorithm(subalgo_type,
                                                  space, **subalgo_hypers)
            elif isinstance(hyper, str) and \
                    hyper.lower() in OptimizationAlgorithm.typenames:
                # pylint: disable=too-many-function-args
                hyper = OptimizationAlgorithm(hyper, space)

            setattr(self, varname, hyper)

    @abstractmethod
    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `metaopt.algo.space.Space`.
        """
        pass

    @abstractmethod
    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Parameters
        ----------
        points : list of tuples of array-likes
           Points from a `metaopt.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        results : list of dicts
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.

        Result
        ------
        objective : numeric
           Evaluation of this problem's objective function.
        gradient : 1D array-like, optional
           Contains values of the derivatives of the `objective` function
           with respect to `params`.
        constraint : list of numeric, optional
           List of constraints expression evaluation which must be greater
           or equal to zero by the problem's definition.

        """
        pass

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return False

    def score(self, point):  # pylint:disable=no-self-use,unused-argument
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        return 0

    def judge(self, point, measurements):  # pylint:disable=no-self-use,unused-argument
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        return None

    @property
    def should_suspend(self):
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
        for attrname in self._hyper_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            if isinstance(attr, BaseAlgorithm):
                attr = attr.configuration
            dict_form[attrname] = attr

        return {self.__class__.__name__.lower(): dict_form}

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance."""
        return self._space

    @space.setter
    def space(self, space_):
        """Propagate changes in defined space to possibly nested algorithms."""
        self._space = space_
        for attr in self.__dict__.values():
            if isinstance(attr, BaseAlgorithm):
                attr.space = space_


# pylint: disable=too-few-public-methods,abstract-method
class OptimizationAlgorithm(BaseAlgorithm, metaclass=Factory):
    """Class used to inject dependency on an algorithm implementation.

    .. seealso:: `Factory` metaclass and `BaseAlgorithm` interface.
    """

    pass
