# pylint:disable=too-many-lines
"""
Search space of optimization problems
=====================================

Classes for representing the search space of an optimization problem.

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
:scipy.stats:`distributions` and should be configured as noted in the
scipy documentation for each specific implementation of a random variable type,
unless noted otherwise!

"""
from __future__ import annotations

import copy
import logging
import numbers
from functools import singledispatch
from typing import Any, Generic, TypeVar

import numpy
from scipy.stats import distributions

from orion.core.utils import float_to_digits_list, format_trials
from orion.core.utils.flatten import flatten

logger = logging.getLogger(__name__)


def check_random_state(seed):
    """Return numpy global rng or RandomState if seed is specified"""
    if seed is None or seed is numpy.random:
        rng = (
            numpy.random.mtrand._rand
        )  # pylint:disable=protected-access,c-extension-no-member
    elif isinstance(seed, numpy.random.RandomState):
        rng = seed
    else:
        try:
            rng = numpy.random.RandomState(seed)
        except Exception as e:
            raise ValueError(
                "%r cannot be used to seed a numpy.random.RandomState"
                " instance" % seed
            ) from e

    return rng


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:  # pylint:disable=too-few-public-methods
    def __repr__(self):
        return "..."


def _to_snake_case(name: str) -> str:
    """Transform a class name ``MyClassName`` to snakecase ``my_class_name``"""
    frags = []

    frag = []
    for c in name:
        if c.isupper() and frag:
            frags.append("".join(frag).lower())
            frag = []

        frag.append(c)

    if frag:
        frags.append("".join(frag).lower())

    return "_".join(frags)


T = TypeVar("T")


class SpaceConverter(Generic[T]):
    """SpaceConverter iterates over an Orion search space.
    This can be used to implement new features for ``orion.algo.space.Space``
    outside of Orion's code base.

    """

    def convert_dimension(self, dimension: Dimension) -> T:
        """Call the dimension conversion handler"""
        return getattr(self, _to_snake_case(type(dimension).__name__))(dimension)

    def dimension(self, dim: Dimension) -> T:
        """Called when the dimension does not have a decicated handler"""

    def real(self, dim: Real) -> T:
        """Called by real dimension"""

    def integer(self, dim: Integer) -> T:
        """Called by integer dimension"""

    def categorical(self, dim: Categorical) -> T:
        """Called by categorical dimension"""

    def fidelity(self, dim: Fidelity) -> T:
        """Called by fidelity dimension"""

    def space(self, space: Space) -> None:
        """Iterate through a research space and visit each dimensions"""
        for _, dim in space.items():
            self.visit(dim)


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
           `scipy.stats.distributions.rv_generic`. Basically,
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
        self._default_value = kwargs.pop("default_value", self.NO_DEFAULT_VALUE)
        self._shape = kwargs.pop("shape", None)
        self.validate()

    def validate(self):
        """Validate dimension arguments"""
        if "random_state" in self._kwargs or "seed" in self._kwargs:
            raise ValueError(
                "random_state/seed cannot be set in a "
                "parameter's definition! Set seed globally!"
            )
        if "discrete" in self._kwargs:
            raise ValueError(
                "Do not use kwarg 'discrete' on `Dimension`, "
                "use pure `_Discrete` class instead!"
            )
        if "size" in self._kwargs:
            raise ValueError("Use 'shape' keyword only instead of 'size'.")

        if (
            self.default_value is not self.NO_DEFAULT_VALUE
            and self.default_value not in self
        ):
            raise ValueError(
                "{} is not a valid value for this Dimension. "
                "Can't set default value.".format(self.default_value)
            )

    def _get_hashable_members(self):
        return (
            self.name,
            self.shape,
            self.type,
            tuple(self._args),
            tuple(self._kwargs.items()),
            self.default_value,
            self._prior_name,
        )

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
        samples = [
            self.prior.rvs(
                *self._args, size=self.shape, random_state=seed, **self._kwargs
            )
            for _ in range(n_samples)
        ]
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
            self.__class__.__name__,
            self.name,
            self._prior_name,
            self._args,
            self._kwargs,
            self.shape,
            self._default_value,
        )

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        args = copy.deepcopy(list(self._args[:]))
        if self._prior_name == "uniform" and len(args) == 2:
            args[1] = args[0] + args[1]
            args[0] = args[0]

        args = list(map(str, args))

        for k, v in self._kwargs.items():
            if isinstance(v, str):
                args += [f"{k}='{v}'"]
            else:
                args += [f"{k}={v}"]

        if self._shape is not None:
            args += [f"shape={self._shape}"]

        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += [f"default_value={repr(self.default_value)}"]

        prior_name = self._prior_name
        if prior_name == "reciprocal":
            prior_name = "loguniform"

        if prior_name == "norm":
            prior_name = "normal"

        return "{prior_name}({args})".format(
            prior_name=prior_name, args=", ".join(args)
        )

    def get_string(self):
        """Build the string corresponding to current dimension"""
        return f"{self.name}~{self.get_prior_string()}"

    @property
    def name(self):
        """See `Dimension` attributes."""
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise TypeError(
                "Dimension's name must be either string or None. "
                "Provided: {}, of type: {}".format(value, type(value))
            )

    @property
    def default_value(self):
        """Return the default value for this dimensions"""
        return self._default_value

    @property
    def type(self):
        """See `Dimension` attributes."""
        return self.__class__.__name__.lower()

    @property
    def prior_name(self):
        """Return the name of the prior"""
        return self._prior_name

    @property
    def shape(self):
        """Return the shape of dimension."""
        # Default shape `None` corresponds to 0-dim (scalar) or shape == ().
        # Read about ``size`` argument in
        # `scipy.stats._distn_infrastructure.rv_generic._argcheck_rvs`
        if self.prior is None:
            return None

        _, _, _, size = self.prior._parse_args_rvs(
            *self._args,  # pylint:disable=protected-access
            size=self._shape,
            **self._kwargs,
        )
        return size

    # pylint:disable=no-self-use
    @property
    def cardinality(self):
        """Return the number of all the possible points from `Dimension`.
        The default value is ``numpy.inf``.
        """
        return numpy.inf


def _is_numeric_array(point):
    """Test whether a point is numerical object or an array containing only numerical objects"""

    def _is_numeric(item):
        return isinstance(item, (numbers.Number, numpy.ndarray))

    try:
        return numpy.all(numpy.vectorize(_is_numeric)(point))
    except TypeError:
        return _is_numeric(point)

    return False


class Real(Dimension):
    """Search space dimension that can take on any real value.

    Parameters
    ----------
    name: str
    prior: str
       See Parameters of `Dimension.__init__()`.
    args: list
    kwargs: dict
       See Parameters of `Dimension.__init__()` for general.

    Notes
    -----
    Real kwargs (extra)

    low: float
       Lower bound (inclusive), optional; default ``-numpy.inf``.
    high: float:
       Upper bound (inclusive), optional; default ``numpy.inf``.
       The upper bound must be inclusive because of rounding errors
       during optimization which may cause values to round exactly
       to the upper bound.
    precision: int
        Precision, optional; default ``4``.
    shape: tuple
       Defines how many dimensions are packed in this `Dimension`.
       Describes the shape of the corresponding tensor.

    """

    def __init__(self, name, prior, *args, **kwargs):
        self._low = kwargs.pop("low", -numpy.inf)
        self._high = kwargs.pop("high", numpy.inf)
        if self._high <= self._low:
            raise ValueError(
                "Lower bound {} has to be less than upper bound {}".format(
                    self._low, self._high
                )
            )

        precision = kwargs.pop("precision", 4)
        if (isinstance(precision, int) and precision > 0) or precision is None:
            self.precision = precision
        else:
            raise TypeError(
                "Precision should be a non-negative int or None, "
                "instead was {} of type {}.".format(precision, type(precision))
            )

        super().__init__(name, prior, *args, **kwargs)

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

        return numpy.all(point_ >= low) and numpy.all(point_ <= high)

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        prior_string = super().get_prior_string()

        if self.precision != 4:
            return prior_string[:-1] + f", precision={self.precision})"

        return prior_string

    def interval(self, alpha=1.0):
        """Return a tuple containing lower and upper bound for parameters.

        If parameters are drawn from an 'open' supported random variable,
        then it will be attempted to calculate the interval from which
        a variable is `alpha`-likely to be drawn from.

        .. note:: Both lower and upper bounds are inclusive.

        """
        prior_low, prior_high = super().interval(alpha)
        return (max(prior_low, self._low), min(prior_high, self._high))

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        .. seealso:: `Dimension.sample`

        """
        samples = []
        for _ in range(n_samples):
            for _ in range(4):
                sample = super().sample(1, seed)
                if sample[0] not in self:
                    nice = False
                    continue
                nice = True
                samples.extend(sample)
                break
            if not nice:
                raise ValueError(
                    "Improbable bounds: (low={}, high={}). "
                    "Please make interval larger.".format(self._low, self._high)
                )

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

    @staticmethod
    def get_cardinality(shape, interval, precision, prior_name):
        """Return the number of all the possible points based and shape and interval"""
        if precision is None or prior_name not in ["loguniform", "reciprocal"]:
            return numpy.inf

        # If loguniform, compute every possible combinations based on precision
        # for each orders of magnitude.

        def format_number(number):
            """Turn number into an array of digits, the size of the precision"""

            formated_number = numpy.zeros(precision)
            digits_list = float_to_digits_list(number)
            length = min(len(digits_list), precision)
            formated_number[:length] = digits_list[:length]

            return formated_number

        min_number = format_number(interval[0])
        max_number = format_number(interval[1])

        # Compute the number of orders of magnitude spanned by lower and upper bounds
        # (if lower and upper bounds on same order of magnitude, span is equal to 1)
        lower_order = numpy.floor(numpy.log10(numpy.abs(interval[0])))
        upper_order = numpy.floor(numpy.log10(numpy.abs(interval[1])))
        order_span = upper_order - lower_order + 1

        # Total number of possibilities for an order of magnitude
        full_cardinality = 9 * 10 ** (precision - 1)

        def num_below(number):

            return (
                numpy.clip(number, a_min=0, a_max=9)
                * 10 ** numpy.arange(precision - 1, -1, -1)
            ).sum()

        # Number of values out of lower bound on lowest order of magnitude
        cardinality_below = num_below(min_number)
        # Number of values out of upper bound on highest order of magnitude.
        # Remove 1 to be inclusive.
        cardinality_above = full_cardinality - num_below(max_number) - 1

        # Full cardinality on all orders of magnitude, minus those out of bounds.
        cardinality = (
            full_cardinality * order_span - cardinality_below - cardinality_above
        )
        return int(cardinality) ** int(numpy.prod(shape) if shape else 1)

    @property
    def cardinality(self):
        """Return the number of all the possible points from Integer `Dimension`"""
        return Real.get_cardinality(
            self.shape, self.interval(), self.precision, self._prior_name
        )


class _Discrete(Dimension):
    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        Discretizes with `numpy.floor` the results from `Dimension.sample`.

        .. seealso:: `Dimension.sample`
        .. seealso:: Discussion in https://github.com/epistimio/orion/issues/56
           if you want to understand better how this `Integer` diamond inheritance
           works.

        """
        samples = super().sample(n_samples, seed)
        # Making discrete by ourselves because scipy does not use **floor**
        return list(map(self.cast, samples))

    def interval(self, alpha=1.0):
        """Return a tuple containing lower and upper bound for parameters.

        If parameters are drawn from an 'open' supported random variable,
        then it will be attempted to calculate the interval from which
        a variable is `alpha`-likely to be drawn from.

        Bounds are integers.

        .. note:: Both lower and upper bounds are inclusive.

        """
        low, high = super().interval(alpha)
        try:
            int_low = int(numpy.floor(low))
        except OverflowError:  # infinity cannot be converted to Python int type
            int_low = -numpy.inf
        try:
            int_high = int(numpy.ceil(high))
        except OverflowError:  # infinity cannot be converted to Python int type
            int_high = numpy.inf
        return (int_low, int_high)

    def __contains__(self, point):
        raise NotImplementedError


class Integer(Real, _Discrete):
    """Search space dimension representing integer values.

    Parameters
    ----------
    name: str
    prior: str
       See Parameters of `Dimension.__init__()`.
    args: list
    kwargs: dict
       See Parameters of `Dimension.__init__()` for general.

    Notes
    -----
    Real kwargs (extra)

    low: float
       Lower bound (inclusive), optional; default ``-numpy.inf``.
    high: float:
       Upper bound (inclusive), optional; default ``numpy.inf``.
    precision: int
        Precision, optional; default ``4``.
    shape: tuple
       Defines how many dimensions are packed in this `Dimension`.
       Describes the shape of the corresponding tensor.

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

        return super().__contains__(point)

    def cast(self, point):
        """Cast a point to int

        If casted point will stay a list or a numpy array depending on the
        given point's type.
        """
        casted_point = numpy.asarray(point).astype(float)

        # Rescale point to make high bound inclusive.
        low, high = self.interval()
        if not numpy.any(numpy.isinf([low, high])):
            high = high - low
            casted_point -= low
            casted_point = casted_point / high
            casted_point = casted_point * (high + (1 - 1e-10))
            casted_point += low
            casted_point = numpy.floor(casted_point).astype(int)
        else:
            casted_point = numpy.floor(casted_point).astype(int)

        if not isinstance(point, numpy.ndarray):
            return casted_point.tolist()

        return casted_point

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        prior_string = super().get_prior_string()
        return prior_string[:-1] + ", discrete=True)"

    @property
    def prior_name(self):
        """Return the name of the prior"""
        return f"int_{super().prior_name}"

    @staticmethod
    def get_cardinality(shape, interval):
        """Return the number of all the possible points based and shape and interval"""
        return int(interval[1] - interval[0] + 1) ** _get_shape_cardinality(shape)

    @property
    def cardinality(self):
        """Return the number of all the possible points from Integer `Dimension`"""
        return Integer.get_cardinality(self.shape, self.interval())


def _get_shape_cardinality(shape):
    """Get the cardinality in a shape which can be int or tuple"""
    shape_cardinality = 1
    if shape is None:
        return shape_cardinality

    if isinstance(shape, int):
        shape = (shape,)

    for cardinality in shape:
        shape_cardinality *= cardinality
    return shape_cardinality


class Categorical(Dimension):
    """Search space dimension that can take on categorical values.

    Parameters
    ----------
    name : str
       See Parameters of `Dimension.__init__()`.
    categories : dict or other iterable
       A dictionary would associate categories to probabilities, else
       it assumes to be drawn uniformly from the iterable.
    kwargs : dict
       See Parameters of `Dimension.__init__()` for general.

    """

    def __init__(self, name, categories, **kwargs):
        if isinstance(categories, dict):
            self.categories = tuple(categories.keys())
            self._probs = tuple(categories.values())
        else:
            self.categories = tuple(categories)
            self._probs = tuple(numpy.tile(1.0 / len(categories), len(categories)))

        # Just for compatibility; everything should be `Dimension` to let the
        # `Transformer` decorators be able to wrap smoothly anything.
        prior = distributions.rv_discrete(
            values=(list(range(len(self.categories))), self._probs)
        )
        super().__init__(name, prior, **kwargs)

    @staticmethod
    def get_cardinality(shape, categories):
        """Return the number of all the possible points based and shape and categories"""
        return len(categories) ** _get_shape_cardinality(shape)

    @property
    def cardinality(self):
        """Return the number of all the possible values from Categorical `Dimension`"""
        return Categorical.get_cardinality(self.shape, self.interval())

    def sample(self, n_samples=1, seed=None):
        """Draw random samples from `prior`.

        .. seealso:: `Dimension.sample`

        """
        rng = check_random_state(seed)
        cat_ndarray = numpy.array(self.categories, dtype=object)
        samples = [
            rng.choice(cat_ndarray, p=self._probs, size=self._shape)
            for _ in range(n_samples)
        ]
        return samples

    def interval(self, alpha=1.0):
        """Return a tuple of possible values that this categorical dimension can take."""
        return self.categories

    def __contains__(self, point):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like

        """
        point_ = numpy.asarray(point, dtype=object)
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

        prior = map(
            lambda x: "{0[0]}: {0[1]:.2f}".format(x)
            if not isinstance(x, _Ellipsis)
            else str(x),
            prior,
        )

        prior = "{" + ", ".join(prior) + "}"

        return "Categorical(name={}, prior={}, shape={}, default value={})".format(
            self.name, prior, self.shape, self.default_value
        )

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        args = list(map(str, self._args[:]))
        args += [f"{k}={v}" for k, v in self._kwargs.items()]
        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += [f"default_value={self.default_value}"]

        cats = [repr(c) for c in self.categories]
        if all(p == self._probs[0] for p in self._probs):
            prior = "[{}]".format(", ".join(cats))
        else:
            probs = list(zip(cats, self._probs))
            prior = "{" + ", ".join(f"{c}: {p:.2f}" for c, p in probs) + "}"

        args = [prior]

        if self._shape is not None:
            args += [f"shape={self._shape}"]
        if self.default_value is not self.NO_DEFAULT_VALUE:
            args += [f"default_value={repr(self.default_value)}"]

        return "choices({args})".format(args=", ".join(args))

    @property
    def get_prior(self):
        """Return the priors"""
        return self._probs

    @property
    def prior_name(self):
        """Return the name of the prior"""
        return "choices"

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
                raise ValueError(f"Invalid category: {value}")

            return categorical_strings[str(value)]

        point_ = numpy.asarray(point, dtype=object)
        cast = numpy.vectorize(get_category, otypes=[object])
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

    Parameters
    ----------
    name : str
        Name of the dimension
    low: int
        Minimum of the fidelity interval.
    high: int
        Maximum of the fidelity interval.
    base: int
        Base logarithm of the fidelity dimension.

    Attributes
    ----------
    name : str
        Name of the dimension
    default_value: int
        Maximum of the fidelity interval.

    """

    # pylint:disable=super-init-not-called
    def __init__(self, name, low, high, base=2):
        if low <= 0:
            raise AttributeError("Minimum resources must be a positive number.")
        elif low > high:
            raise AttributeError(
                "Minimum resources must be smaller than maximum resources."
            )
        if base < 1:
            raise AttributeError("Base should be greater than or equal to 1")
        self.name = name
        self.low = int(low)
        self.high = int(high)
        self.base = int(base)
        self.prior = None
        self._prior_name = "None"

    @property
    def default_value(self):
        """Return `high`"""
        return self.high

    @staticmethod
    def get_cardinality(shape, interval):
        """Return cardinality of Fidelity dimension, leave it to 1 as Fidelity dimension
        does not contribute to cardinality in a fixed way now.
        """
        return 1

    @property
    def cardinality(self):
        """Return cardinality of Fidelity dimension, leave it to 1 as Fidelity dimension
        does not contribute to cardinality in a fixed way now.
        """
        return Fidelity.get_cardinality(self.shape, self.interval())

    def get_prior_string(self):
        """Build the string corresponding to current prior"""
        args = [str(self.low), str(self.high)]

        if self.base != 2:
            args += [f"base={self.base}"]

        return "fidelity({})".format(", ".join(args))

    def validate(self):
        """Do not do anything."""
        raise NotImplementedError

    def sample(self, n_samples=1, seed=None):
        """Do not do anything."""
        return [self.high for i in range(n_samples)]

    def interval(self, alpha=1.0):
        """Do not do anything."""
        return (self.low, self.high)

    def cast(self, point=0):
        """Do not do anything."""
        raise NotImplementedError

    def __repr__(self):
        """Represent the object as a string."""
        return "{}(name={}, low={}, high={}, base={})".format(
            self.__class__.__name__, self.name, self.low, self.high, self.base
        )

    def __contains__(self, value):
        """Check if constraints hold for this `point` of `Dimension`.

        :param point: a parameter corresponding to this `Dimension`.
        :type point: numeric or array-like
        """
        return self.low <= value <= self.high


class Space(dict):
    """Represents the search space.

    It is a sorted dictionary which contains `Dimension` objects.
    The dimensions are sorted based on their names.
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
        trials: list of `orion.core.worker.trial.Trial`
           Each element is a separate sample of this space, a trial containing
           values associated with the corresponding dimension.

        """
        rng = check_random_state(seed)
        samples = [dim.sample(n_samples, rng) for dim in self.values()]
        return [format_trials.tuple_to_trial(point, self) for point in zip(*samples)]

    def interval(self, alpha=1.0):
        """Return a list with the intervals for each contained dimension."""
        res = list()
        for dim in self.values():
            if dim.type == "categorical":
                res.append(dim.categories)
            else:
                res.append(dim.interval(alpha))
        return res

    def __getitem__(self, key):
        """Wrap __getitem__ to allow searching with position."""
        if isinstance(key, str):
            return super().__getitem__(key)

        values = list(self.values())
        return values[key]

    def __setitem__(self, key, value):
        """Wrap __setitem__ to allow only ``Space.contains`` class, e.g. `Dimension`,
        values and string keys.
        """
        if not isinstance(key, str):
            raise TypeError(
                "Keys registered to {} must be string types. "
                "Provided: {}".format(self.__class__.__name__, key)
            )
        if not isinstance(value, self.contains):
            raise TypeError(
                "Values registered to {} must be {} types. "
                "Provided: {}".format(
                    self.__class__.__name__, self.contains.__name__, value
                )
            )
        if key in self:
            raise ValueError(
                "There is already a Dimension registered with this name. "
                "Register it with another name. Provided: {}".format(key)
            )
        super().__setitem__(key, value)

    def __contains__(self, key_or_trial):
        """Check whether `trial` is within the bounds of the space.
        Or check if a name for a dimension is registered in this space.

        Parameters
        ----------
        key_or_trial: str or `orion.core.worker.trial.Trial`
            If str, test if the string is a dimension part of the search space.
            If a Trial, test if trial's hyperparameters fit the current search space.
        """
        if isinstance(key_or_trial, str):
            return super().__contains__(key_or_trial)

        trial = key_or_trial
        flattened_params = flatten(trial.params)
        keys = set(flattened_params.keys())
        for dim_name, dim in self.items():
            if dim_name not in keys or flattened_params[dim_name] not in dim:
                return False

            keys.remove(dim_name)

        return len(keys) == 0

    def __repr__(self):
        """Represent as a string the space and the dimensions it contains."""
        dims = list(self.values())
        return "Space([{}])".format(",\n       ".join(map(str, dims)))

    def items(self):
        """Return items sorted according to keys"""
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        """Return values sorted according to keys"""
        return [self[k] for k in self.keys()]

    def keys(self):
        """Return sorted keys"""
        return list(iter(self))

    def __iter__(self):
        """Return sorted keys"""
        return iter(sorted(super().keys()))

    @property
    def configuration(self):
        """Return a dictionary of priors."""
        return {name: dim.get_prior_string() for name, dim in self.items()}

    @property
    def cardinality(self):
        """Return the number of all all possible sets of samples in the space"""
        capacities = 1
        for dim in self.values():
            capacities *= dim.cardinality
        return capacities


@singledispatch
def to_orionspace(space: Any) -> Space:
    """Convert a third party search space into an Orion compatible space

    Raises
    ------
    NotImplementedError if no conversion was registered
    """
    raise NotImplementedError()
