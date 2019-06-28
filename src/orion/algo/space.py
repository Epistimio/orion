# -*- coding: utf-8 -*-
# pylint:disable=too-many-lines
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
import numbers

import numpy
from scipy.stats import distributions


def check_random_state(seed):
    """Return numpy global rng or RandomState if seed is specified"""
    if seed is None:
        return numpy.random.mtrand._rand  # pylint:disable=protected-access,c-extension-no-member

    return numpy.random.RandomState(seed)


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:  # pylint:disable=too-few-public-methods
    def __repr__(self):
        return '...'


class Dimension:
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

    NO_DEFAULT_VALUE = None

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

        if isinstance(prior, str):
            self._prior_name = prior
            self.prior = getattr(distributions, prior)
        elif prior is None:
            self._prior_name = "None"
            self.prior = prior
        else:
            self._prior_name = prior.name
            self.prior = prior
        self._args = args
        self._kwargs = kwargs
        self._default_value = kwargs.pop('default_value', self.NO_DEFAULT_VALUE)
        self._shape = kwargs.pop('shape', None)
        self.validate()

    def validate(self):
        """Validate dimension arguments"""
        if 'random_state' in self._kwargs or 'seed' in self._kwargs:
            raise ValueError("random_state/seed cannot be set in a "
                             "parameter's definition! Set seed globally!")
        if 'discrete' in self._kwargs:
            raise ValueError("Do not use kwarg 'discrete' on `Dimension`, "
                             "use pure `_Discrete` class instead!")
        if 'size' in self._kwargs:
            raise ValueError("Use 'shape' keyword only instead of 'size'.")

        if self.default_value is not self.NO_DEFAULT_VALUE and self.default_value not in self:
            raise ValueError("{} is not a valid value for this Dimension. "
                             "Can't set default value.".format(self.default_value))

    def _get_hashable_members(self):
        return (self.name, self.shape, self.type, tuple(self._args), tuple(self._kwargs.items()),
                self.default_value, self._prior_name)

    # pylint:disable=protected-access
    def __eq__(self, other):
        """Return True if other is the same dimension as self"""
        if not isinstance(other, Dimension):
            return False

        return self._get_hashable_members() == other._get_hashable_members()

    def __hash__(self):
        """Return the hash of the hashable members"""
        return hash(self._get_hashable_members())

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
        samples = [self.prior.rvs(*self._args, size=self.shape,
                                  random_state=seed,
                                  **self._kwargs) for _ in range(n_samples)]
        return samples

    def cast(self, point):
        """Cast a point to dimension's type

        If casted point will stay a list or a numpy array depending on the
        given point's type.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def __repr__(self):
        """Represent the object as a string."""
        return "{0}(name={1}, prior={{{2}: {3}, {4}}}, shape={5}, default value={6})".format(
            self.__class__.__name__, self.name, self._prior_name,
            self._args, self._kwargs, self.shape, self._default_value)

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        args = list(map(str, self._args[:]))
        args += ["{}={}".format(k, v) for k, v in self._kwargs.items()]
        if self._shape is not None:
            args += ['shape={}'.format(self._shape)]
        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += ['default_value={}'.format(repr(self.default_value))]
        return "{prior_name}({args})".format(prior_name=self._prior_name, args=", ".join(args))

    def get_string(self):
        """Build the string corresponding to current dimension"""
        return "{name}~{prior}".format(name=self.name, prior=self.get_prior_string())

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
    def default_value(self):
        """Return the default value for this dimensions"""
        return self._default_value

    @property
    def type(self):
        """See `Dimension` attributes."""
        return self.__class__.__name__.lower()

    @property
    def shape(self):
        """Return the shape of dimension."""
        # Default shape `None` corresponds to 0-dim (scalar) or shape == ().
        # Read about ``size`` argument in
        # `scipy.stats._distn_infrastructure.rv_generic._argcheck_rvs`
        if self.prior is None:
            return None

        _, _, _, size = self.prior._parse_args_rvs(*self._args,  # pylint:disable=protected-access
                                                   size=self._shape,
                                                   **self._kwargs)
        return size


def _is_numeric_array(point):
    """Test whether a point is numerical object or an array containing only numerical objects"""
    def _is_numeric(item):
        return isinstance(item, (numbers.Number, numpy.ndarray))

    try:
        return all(_is_numeric(item) for item in point)
    except TypeError:
        return _is_numeric(point)

    return False


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
            raise ValueError("Lower bound {} has to be less than upper bound {}"
                             .format(self._low, self._high))

        super(Real, self).__init__(name, prior, *args, **kwargs)

    def __contains__(self, point):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like

        .. note:: Default `Dimension` does not have any extra constraints.
           It just checks whether point lies inside the support and the shape.

        """
        if not _is_numeric_array(point):
            return False

        low, high = self.interval()

        point_ = numpy.asarray(point)

        if point_.shape != self.shape:
            return False

        return numpy.all(point_ < high) and numpy.all(point_ >= low)

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

    # pylint:disable=no-self-use
    def cast(self, point):
        """Cast a point to float

        If casted point will stay a list or a numpy array depending on the
        given point's type.
        """
        casted_point = numpy.asarray(point).astype(float)

        if not isinstance(point, numpy.ndarray):
            return casted_point.tolist()

        return casted_point


class _Discrete(Dimension):

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        Discretizes with `numpy.floor` the results from `Dimension.sample`.

        .. seealso:: `Dimension.sample`
        .. seealso:: Discussion in https://github.com/epistimio/orion/issues/56
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

    def __contains__(self, point):
        raise NotImplementedError


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
        if not _is_numeric_array(point):
            return False

        point_ = numpy.asarray(point)
        if not numpy.all(numpy.equal(numpy.mod(point_, 1), 0)):
            return False

        return super(Integer, self).__contains__(point)

    # pylint:disable=no-self-use
    def cast(self, point):
        """Cast a point to int

        If casted point will stay a list or a numpy array depending on the
        given point's type.
        """
        casted_point = numpy.asarray(point).astype(int)

        if not isinstance(point, numpy.ndarray):
            return casted_point.tolist()

        return casted_point


class Categorical(Dimension):
    """Subclass of `Dimension` for representing categorical parameters.

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

        return "Categorical(name={0}, prior={1}, shape={2}, default value={3})"\
               .format(self.name, prior, self.shape, self.default_value)

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        args = list(map(str, self._args[:]))
        args += ["{}={}".format(k, v) for k, v in self._kwargs.items()]
        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += ['default_value={}'.format(self.default_value)]

        cats = [repr(c) for c in self.categories]
        if all(p == self._probs[0] for p in self._probs):
            prior = '[{}]'.format(", ".join(cats))
        else:
            probs = list(zip(cats, self._probs))
            prior = '{' + ", ".join('{0}: {1:.2f}'.format(c, p) for c, p in probs) + '}'

        args = [prior]

        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += ['default_value={}'.format(repr(self.default_value))]

        return 'choices({args})'.format(args=', '.join(args))

    def cast(self, point):
        """Cast a point to some category

        Casted point will stay a list or a numpy array depending on the
        given point's type.

        Raises
        ------
        ValueError
            If one of the category in `point` is not present in current Categorical Dimension.

        """
        categorical_strings = {str(c): c for c in self.categories}

        def get_category(value):
            """Return category corresponding to a string else return singleton object"""
            if str(value) not in categorical_strings:
                raise ValueError("Invalid category: {}".format(value))

            return categorical_strings[str(value)]

        point_ = numpy.asarray(point, dtype=numpy.object)
        cast = numpy.vectorize(get_category, otypes=[numpy.object])
        casted_point = cast(point_)

        if not isinstance(point, numpy.ndarray):
            return casted_point.tolist()

        return casted_point


class Fidelity(Dimension):
    """Fidelity `Dimension` for representing multi-fidelity.

    Fidelity dimensions are not optimized by the algorithms. If it supports multi-fidelity, the
    algorithm will select a fidelity level for which it will sample hyper-parameter values to
    explore a low fidelity space. This class is used as a place-holder so that algorithms can
    discern fidelity dimensions from hyper-parameter dimensions.

    Attributes
    ----------
    name : str
    type : str

    """

    # pylint:disable=super-init-not-called
    def __init__(self, name):
        """Fidelity dimension that can represent a fidelity level.

        Parameters
        ----------
        name : str

        """
        self.name = name
        self.prior = None
        self._prior_name = 'None'

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        return 'fidelity()'

    def validate(self):
        """Do not do anything."""
        raise NotImplementedError

    def sample(self, n_samples=1, seed=None):
        """Do not do anything."""
        return ['fidelity']

    def interval(self, alpha=1.0):
        """Do not do anything."""
        raise NotImplementedError

    def cast(self, point=0):
        """Do not do anything."""
        raise NotImplementedError

    def __repr__(self):
        """Represent the object as a string."""
        return "{0}(name={1})".format(self.__class__.__name__, self.name)

    def __contains__(self, value):
        """Check if constraints hold for this `point` of `Dimension`.

        .. note ::

            Always True for Fidelity.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like
        """
        return True


class Space(OrderedDict):
    """Represents the search space.

    It is an ordered dictionary which :attr:`contains` `Dimension` objects.
    That class attribute is used to perform checks on :meth:`register`.
    """

    contains = Dimension

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
        """Wrap __setitem__ to allow only ``Space.contains`` class, e.g. `Dimension`,
        values and string keys.
        """
        if not isinstance(key, str):
            raise TypeError("Keys registered to {} must be string types. "
                            "Provided: {}".format(self.__class__.__name__, key))
        if not isinstance(value, self.contains):
            raise TypeError("Values registered to {} must be {} types. "
                            "Provided: {}".format(self.__class__.__name__,
                                                  self.contains.__name__, value))
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


def pack_point(point, space):
    """Take a list of points and pack it appropriately as a point from `space`.

    :param point: array-like or list of numbers
    :param space: problem's parameter definition,
       instance of `orion.algo.space.Space`

    .. note:: It works only if dimensions included in `space` have 0D or 1D shape.

    :returns: list of numbers or tuples
    """
    packed = []
    idx = 0
    for dim in space.values():
        shape = dim.shape
        if shape:
            assert len(shape) == 1
            next_idx = idx + shape[0]
            packed.append(tuple(point[idx:next_idx]))
            idx = next_idx
        else:
            packed.append(point[idx])
            idx += 1
    assert packed in space
    return packed


def unpack_point(point, space):
    """Flatten `point` in `space` and convert it to a 1D `numpy.ndarray`.

    :param point: list of number or tuples, in `space`
    :param space: problem's parameter definition,
       instance of `orion.algo.space.Space`

    .. note:: It works only if dimensions included in `space` have 0D or 1D shape.

    :returns: a list of float numbers
    """
    unpacked = []
    for subpoint, dim in zip(point, space.values()):
        shape = dim.shape
        if shape:
            assert len(shape) == 1
            unpacked.extend(subpoint)
        else:
            unpacked.append(subpoint)
    return unpacked
