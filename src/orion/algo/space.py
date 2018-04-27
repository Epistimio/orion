# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.space` -- Objects describing a problem's domain
==================================================================

.. module:: space
   :platform: Unix
   :synopsis: Classes for representing the search space of an
      optimization problem.

There are 3 classes representing possible parameter types. All of them subclass
the base class `Dimension`:

    * `Real`
    * `Integer`
    * `Categorical`

These are instantiated to declare a problem's parameter space. Oríon registers
them in a ordered dictionary, `Space`, which describes how the parameters should
be in order for `orion.algo.base.AbstractAlgorithm` implementations to
communicate with `orion.core`.

Parameter values recorded in `orion.core.worker.trial.Trial` objects must be
and are in concordance with `orion.algo.space` objects. These objects will be
defined by `orion.core` using the user script's configuration file.

Prior distributions, contained in `Dimension` classes, are based on
`scipy.stats.distributions` and should be configured as noted in the
scipy documentation for each specific implentation of a random variable type,
unless noted otherwise!

"""

from collections import OrderedDict

import numpy
from scipy._lib._util import check_random_state
from scipy.stats import distributions


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:  # pylint:disable=too-few-public-methods
    def __repr__(self):
        return '...'


class Dimension(object):
    """Base class for search space dimensions.

    Attributes
    ----------
    name : str
       Unique identifier for this `Dimension`.
    type : str
       Identifier for the type of parameters this `Dimension` is representing.
       it can be 'real', 'integer', or 'categorical' (name of a subclass).
    prior : `scipy.stats.distributions.rv_generic`
       A distribution over the original dimension.
    shape : tuple
       Defines how many dimensions are packed in this `Dimension`.
       Describes the shape of the corresponding tensor.

    """

    def __init__(self, name, prior, *args, **kwargs):
        """Init code which is common for `Dimension` subclasses.

        Parameters
        ----------
        name : str
           Unique identifier associated with this `Dimension`,
           e.g. 'learning_rate'.
        prior : str | `scipy.stats.distributions.rv_generic`
           Corresponds to a name of an instance or an instance itself of
           `scipy.stats._distn_infrastructure.rv_generic`. Basically,
           the name of the distribution one wants to use as a :attr:`prior`.
        args : list
        kwargs : dict
           Shape parameter(s) for the `prior` distribution.
           Should include all the non-optional arguments.
           It may include ``loc``, ``scale``, ``shape``.

        .. seealso:: `scipy.stats.distributions` for possible values of
           `prior` and their arguments.

        """
        self._name = None
        self.name = name
        if 'random_state' in kwargs or 'seed' in kwargs:
            raise ValueError("random_state/seed cannot be set in a "
                             "parameter's definition! Set seed globally!")
        if 'discrete' in kwargs:
            raise ValueError("Do not use kwarg 'discrete' on `Dimension`, "
                             "use pure `_Discrete` class instead!")
        if isinstance(prior, str):
            self._prior_name = prior
            self.prior = getattr(distributions, prior)
        else:
            self._prior_name = prior.name
            self.prior = prior
        self._args = args
        self._kwargs = kwargs
        # Default shape `None` corresponds to 0-dim (scalar) or shape == ().
        # Read about ``size`` argument in
        # `scipy.stats._distn_infrastructure.rv_generic._argcheck_rvs`
        if 'size' in kwargs:
            raise ValueError("Use 'shape' keyword only instead of 'size'.")
        self._shape = self._kwargs.pop('shape', None)

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        Parameters
        ----------
        n_samples : int, optional
           The number of samples to be drawn. Default is 1 sample.
        seed : None | int | ``numpy.random.RandomState`` instance, optional
           This parameter defines the RandomState object to use for drawing
           random variates. If None (or np.random), the **global**
           np.random state is used. If integer, it is used to seed a
           RandomState instance **just for the call of this function**.
           Default is None.

           Set random state to something other than None for reproducible
           results.

        .. warning:: Setting `seed` with an integer will cause the same ndarray
           to be sampled if ``n_samples > 0``. Set `seed` with a
           ``numpy.random.RandomState`` to carry on the changes in random state
           across many samples.

        """
        samples = [self.prior.rvs(*self._args, size=self._shape,
                                  random_state=seed,
                                  **self._kwargs) for _ in range(n_samples)]
        return samples

    def interval(self, alpha=1.0):
        """Return a tuple containing lower and upper bound for parameters.

        If parameters are drawn from an 'open' supported random variable,
        then it will be attempted to calculate the interval from which
        a variable is `alpha`-likely to be drawn from.

        .. note:: Lower bound is inclusive, upper bound is exclusive.

        """
        return self.prior.interval(alpha, *self._args, **self._kwargs)

    def __contains__(self, point):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like

        .. note:: Default `Dimension` does not have any extra constraints.
           It just checks whether point lies inside the support and the shape.

        """
        low, high = self.interval()
        point_ = numpy.asarray(point)
        if point_.shape != self.shape:
            return False
        return numpy.all(point_ < high) and numpy.all(point_ >= low)

    def __repr__(self):
        """Represent the object as a string."""
        return "{0}(name={1}, prior={{{2}: {3}, {4}}}, shape={5})".format(
            self.__class__.__name__, self.name, self._prior_name,
            self._args, self._kwargs, self.shape)

    @property
    def name(self):
        """See `Dimension` attributes."""
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise TypeError("Dimension's name must be either string or None. "
                            "Provided: {}, of type: {}".format(value, type(value)))

    @property
    def type(self):
        """See `Dimension` attributes."""
        return self.__class__.__name__.lower()

    @property
    def shape(self):
        """Return the shape of dimension."""
        _, _, _, size = self.prior._parse_args_rvs(*self._args,  # pylint:disable=protected-access
                                                   size=self._shape,
                                                   **self._kwargs)
        return size


class Real(Dimension):
    """Subclass of `Dimension` for representing real parameters.

    Attributes
    ----------
    name : str
    type : str
    prior : `scipy.stats.distributions.rv_generic`
    shape : tuple
       See Attributes of `Dimension`.
    low : float
       Constrain with a lower bound (inclusive), default ``-numpy.inf``.
    high : float
       Constrain with an upper bound (exclusive), default ``numpy.inf``.

    """

    def __init__(self, name, prior, *args, **kwargs):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        name : str
        prior : str
           See Parameters of `Dimension.__init__`.
        args : list
        kwargs : dict
           See Parameters of `Dimension.__init__` for general.

        Real kwargs (extra)
        -------------------
        low : float
           Lower bound (inclusive), optional; default ``-numpy.inf``.
        high : float:
           Upper bound (exclusive), optional; default ``numpy.inf``.

        """
        self._low = kwargs.pop('low', -numpy.inf)
        self._high = kwargs.pop('high', numpy.inf)
        if self._high <= self._low:
            raise ValueError("Lower bound {} has to be less "
                             "than upper bound {}".format(self._low, self._high))
        super(Real, self).__init__(name, prior, *args, **kwargs)

    def interval(self, alpha=1.0):
        """Return a tuple containing lower and upper bound for parameters.

        If parameters are drawn from an 'open' supported random variable,
        then it will be attempted to calculate the interval from which
        a variable is `alpha`-likely to be drawn from.

        .. note:: Lower bound is inclusive, upper bound is exclusive.

        """
        prior_low, prior_high = super(Real, self).interval(alpha)
        return (max(prior_low, self._low), min(prior_high, self._high))

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        .. seealso:: `Dimension.sample`

        """
        samples = []
        for _ in range(n_samples):
            for _ in range(4):
                sample = super(Real, self).sample(1, seed)
                if sample[0] not in self:
                    nice = False
                    continue
                nice = True
                samples.extend(sample)
                break
            if not nice:
                raise ValueError("Improbable bounds: (low={0}, high={1}). "
                                 "Please make interval larger.".format(self._low, self._high))

        return samples


class _Discrete(Dimension):

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        Discretizes with `numpy.floor` the results from `Dimension.sample`.

        .. seealso:: `Dimension.sample`
        .. seealso:: Discussion in https://github.com/mila-udem/orion/issues/56
           if you want to understand better how this `Integer` diamond inheritance
           works.

        """
        samples = super(_Discrete, self).sample(n_samples, seed)
        # Making discrete by ourselves because scipy does not use **floor**
        return list(map(lambda x: numpy.floor(x).astype(int), samples))

    def interval(self, alpha=1.0):
        """Return a tuple containing lower and upper bound for parameters.

        If parameters are drawn from an 'open' supported random variable,
        then it will be attempted to calculate the interval from which
        a variable is `alpha`-likely to be drawn from.

        Bounds are integers.

        .. note:: Lower bound is inclusive, upper bound is exclusive.

        """
        low, high = super(_Discrete, self).interval(alpha)
        try:
            int_low = int(numpy.floor(low))
        except OverflowError:  # infinity cannot be converted to Python int type
            int_low = -numpy.inf
        try:
            int_high = int(numpy.floor(high))
        except OverflowError:  # infinity cannot be converted to Python int type
            int_high = numpy.inf
        if int_high < high:  # Exclusive upper bound
            int_high += 1
        return (int_low, int_high)


class Integer(Real, _Discrete):
    """Subclass of `Dimension` for representing integer parameters.

    Attributes
    ----------
    name : str
    type : str
    prior : `scipy.stats.distributions.rv_generic`
    shape : tuple
       See Attributes of `Dimension`.

    """

    def __contains__(self, point):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like

        `Integer` will check whether `point` contains only integers.

        """
        point_ = numpy.asarray(point)
        if not numpy.all(numpy.equal(numpy.mod(point_, 1), 0)):
            return False
        return super(Integer, self).__contains__(point)


class Categorical(Dimension):
    """Subclass of `Dimension` for representing integer parameters.

    Attributes
    ----------
    name : str
    type : str
    prior : `scipy.stats.distributions.rv_generic`
    shape : tuple
       See Attributes of `Dimension`.
    categories : tuple
       A set of unordered stuff to pick out from, except if enum

    """

    def __init__(self, name, categories, **kwargs):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        name : str
           See Parameters of `Dimension.__init__`.
        categories : dict or other iterable
           A dictionary would associate categories to probabilities, else
           it assumes to be drawn uniformly from the iterable.
        kwargs : dict
           See Parameters of `Dimension.__init__` for general.

        """
        if isinstance(categories, dict):
            self.categories = tuple(categories.keys())
            self._probs = tuple(categories.values())
        else:
            self.categories = tuple(categories)
            self._probs = tuple(numpy.tile(1. / len(categories), len(categories)))

        # Just for compatibility; everything should be `Dimension` to let the
        # `Transformer` decorators be able to wrap smoothly anything.
        prior = distributions.rv_discrete(values=(list(range(len(self.categories))),
                                                  self._probs))
        super(Categorical, self).__init__(name, prior, **kwargs)

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        .. seealso:: `Dimension.sample`

        """
        rng = check_random_state(seed)
        cat_ndarray = numpy.array(self.categories, dtype=numpy.object)
        samples = [rng.choice(cat_ndarray, p=self._probs, size=self._shape)
                   for _ in range(n_samples)]
        return samples

    def interval(self, alpha=1.0):
        """Return a tuple of possible values that this categorical dimension
        can take.

        .. warning:: This method makes no sense for categorical variables. Use
           ``self.categories`` instead.

        """
        raise RuntimeError("Categories have no ``interval`` (as they are not ordered).\n"
                           "Use ``self.categories`` instead.")

    def __contains__(self, point):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like

        """
        point_ = numpy.asarray(point, dtype=numpy.object)
        if point_.shape != self.shape:
            return False
        _check = numpy.vectorize(lambda x: x in self.categories)
        return numpy.all(_check(point_))

    def __repr__(self):
        """Represent the object as a string."""
        if len(self.categories) > 5:
            cats = self.categories[:2] + self.categories[-2:]
            probs = self._probs[:2] + self._probs[-2:]
            prior = list(zip(cats, probs))
            prior.insert(2, _Ellipsis())
        else:
            cats = self.categories
            probs = self._probs
            prior = list(zip(cats, probs))

        prior = map(lambda x: '{0[0]}: {0[1]:.2f}'.format(x)
                    if not isinstance(x, _Ellipsis) else str(x), prior)
        prior = "{" + ', '.join(prior) + "}"

        return "Categorical(name={0}, prior={1}, shape={2})".format(self.name,
                                                                    prior,
                                                                    self.shape)


class Space(OrderedDict):
    """Represents the search space."""

    def register(self, dimension):
        """Register a new dimension to `Space`."""
        self[dimension.name] = dimension

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from this space.

        Parameters
        ----------
        n_samples : int, optional
           The number of samples to be drawn. Default is 1 sample.
        seed : None | int | ``numpy.random.RandomState`` instance, optional
           This parameter defines the RandomState object to use for drawing
           random variates. If None (or np.random), the **global**
           np.random state is used. If integer, it is used to seed a
           RandomState instance **just for the call of this function**.
           Default is None.

           Set random state to something other than None for reproducible
           results.

        Returns
        -------
        points : list of tuples of array-likes
           Each element is a separate sample of this space, a list containing
           values associated with the corresponding dimension. Values are in the
           same order as the contained dimensions. Their shape is determined
           by ``dimension.shape``.

        """
        rng = check_random_state(seed)
        samples = [dim.sample(n_samples, rng) for dim in self.values()]
        return list(zip(*samples))

    def interval(self, alpha=1.0):
        """Return a list with the intervals for each contained dimension.

        .. note:: Lower bound is inclusive, upper bound is exclusive.

        """
        res = list()
        for dim in self.values():
            if dim.type == 'categorical':
                res.append(dim.categories)
            else:
                res.append(dim.interval(alpha))
        return res

    def __getitem__(self, key):
        """Wrap __getitem__ to allow searching with position."""
        if isinstance(key, str):
            return super(Space, self).__getitem__(key)

        values = list(self.values())
        return values[key]

    def __setitem__(self, key, value):
        """Wrap __setitem__ to allow only `Dimension`s values and string keys."""
        if not isinstance(key, str):
            raise TypeError("Keys registered to Space must be string types. "
                            "Provided: {}".format(key))
        if not isinstance(value, Dimension):
            raise TypeError("Values registered to Space must be Dimension types. "
                            "Provided: {}".format(value))
        if key in self:
            raise ValueError("There is already a Dimension registered with this name. "
                             "Register it with another name. Provided: {}".format(key))
        super(Space, self).__setitem__(key, value)

    def __contains__(self, value):
        """Check whether `value` is within the bounds of the space.
        Or check if a name for a dimension is registered in this space.

        :param value: list of values associated with the dimensions contained
           or a string indicating a dimension's name.

        """
        if isinstance(value, str):
            return super(Space, self).__contains__(value)

        try:
            len(value)
        except TypeError as exc:
            raise TypeError("Can check only for dimension names or "
                            "for tuples with parameter values.") from exc

        if not self:
            return False

        for component, dim in zip(value, self.values()):
            if component not in dim:
                return False

        return True

    def __repr__(self):
        """Represent as a string the space and the dimensions it contains."""
        dims = list(self.values())
        return "Space([{}])".format(',\n       '.join(map(str, dims)))
