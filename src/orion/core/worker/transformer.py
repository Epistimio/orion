# pylint: disable=too-many-lines
"""
Perform transformations on Dimensions
=====================================

Provide functions and classes to build a Space which an algorithm can operate on.

"""
import copy
import functools
import itertools
from abc import ABCMeta, abstractmethod

import numpy

from orion.algo.space import Categorical, Dimension, Fidelity, Integer, Real, Space
from orion.core.utils import format_trials
from orion.core.utils.flatten import flatten

NON_LINEAR = ["loguniform", "reciprocal"]


# pylint: disable=unused-argument
@functools.singledispatch
def build_transform(dim, type_requirement, dist_requirement):
    """Base transformation factory

    Parameters
    ----------
    dim: `orion.algo.space.Dimension`
        A dimension object which may need transformations to match provided requirements.
    type_requirement: str, None
        String defining the requirement of the algorithm. It can be one of the following
        - 'real', the dim should be transformed so type is `orion.algo.space.Real`
        - 'integer', the dim should be transformed so type is `orion.algo.space.Integer`
        - 'numerical', the dim should be transformed so type is either `orion.algo.space.Integer` or
        `orion.algo.space.Real`
        - None, no requirement
    dist_requirement: str, None
        String defining the distribution requirement of the algorithm.
        - 'linear', any dimension with logarithmic prior while be linearized
        - None, no requirement

    """
    return []


@build_transform.register(Categorical)
def _(dim, type_requirement, dist_requirement):
    transformers = []
    if type_requirement == "real":
        transformers.extend(
            [Enumerate(dim.categories), OneHotEncode(len(dim.categories))]
        )
    elif type_requirement in ["integer", "numerical"]:
        transformers.append(Enumerate(dim.categories))

    return transformers


@build_transform.register(Fidelity)
def _(dim, type_requirement, dist_requirement):
    return []


@build_transform.register(Integer)
def _(dim, type_requirement, dist_requirement):
    transformers = []
    if dist_requirement == "linear" and dim.prior_name[4:] in NON_LINEAR:
        transformers.extend([Reverse(Quantize()), Linearize()])
        # NOTE: we do not turn back to integer even though linearize outputs real
        #       otherwise the mapping from exp(int) to int squashes out lots of possible values.
    elif type_requirement == "real":
        transformers.append(Reverse(Quantize()))

    return transformers


@build_transform.register(Real)
def _(dim, type_requirement, dist_requirement):
    transformers = []
    if dim.precision is not None:
        transformers.append(Precision(dim.precision))

    if dist_requirement == "linear" and dim.prior_name in NON_LINEAR:
        transformers.append(Linearize())
    elif type_requirement == "integer":
        # NOTE: This may cause out-of-bound errors for rounded reals. Not fixed for now
        #       because there are no foreseeable algorithms that may require integer type.
        transformers.append(Quantize())

    return transformers


def transform(original_space, type_requirement, dist_requirement):
    """Build a transformed space"""
    space = TransformedSpace(original_space)
    for dim in original_space.values():
        transformers = build_transform(dim, type_requirement, dist_requirement)
        space.register(
            TransformedDimension(
                transformer=Compose(transformers, dim.type), original_dimension=dim
            )
        )

    return space


def reshape(space, shape_requirement):
    """Build a reshaped space"""
    if shape_requirement is None:
        return space

    # We assume shape_requirement == 'flattened'

    reshaped_space = ReshapedSpace(space)

    for dim_index, dim in enumerate(space.values()):
        if not dim.shape:
            reshaped_space.register(
                ReshapedDimension(
                    transformer=Identity(dim.type),
                    original_dimension=dim,
                    index=dim_index,
                )
            )
        else:
            for index in itertools.product(*map(range, dim.shape)):
                key = f'{dim.name}[{",".join(map(str, index))}]'
                reshaped_space.register(
                    ReshapedDimension(
                        transformer=View(dim.shape, index, dim.type),
                        original_dimension=dim,
                        name=key,
                        index=dim_index,
                    )
                )

    return reshaped_space


def build_required_space(
    original_space, type_requirement=None, shape_requirement=None, dist_requirement=None
):
    """Build a :class:`orion.algo.space.Space` object which agrees to the `requirements` imposed
    by the desired optimization algorithm.

    It uses appropriate cascade of `Transformer` objects per `orion.algo.space.Dimension`
    contained in `original_space`. `ReshapedTransformer` objects are used above
    the `Transformer` if the optimizatios algorithm requires flattened dimensions.

    Parameters
    ----------
    original_space : `orion.algo.space.Space`
        Original problem's definition of parameter space given by the user to Oríon.
    type_requirement: str, None
        String defining the requirement of the algorithm. It can be one of the following
        - 'real', the dim should be transformed so type is `orion.algo.space.Real`
        - 'integer', the dim should be transformed so type is `orion.algo.space.Integer`
        - 'numerical', the dim should be transformed so type is either `orion.algo.space.Integer` or
        `orion.algo.space.Real`
        - None, no requirement
    shape_requirement: str, None
        String defining the shape requirement of the algorithm.
        - 'flattened', any dimension with shape > 1 will be flattened
        - None, no requirement
    dist_requirement: str, None
        String defining the distribution requirement of the algorithm.
        - 'linear', any dimension with logarithmic prior while be linearized
        - None, no requirement

    """
    space = transform(original_space, type_requirement, dist_requirement)
    space = reshape(space, shape_requirement)

    return space


class Transformer(metaclass=ABCMeta):
    """Define an (injective) function and its inverse. Base transformation class.

    Attributes
    ----------
    target_type: str
        Defines the type of the target space of the forward function.
        It can provide one of the values: ``['real', 'integer', 'categorical']``.
    domain_type: str
        Is similar to ``target_type`` but it refers to the domain.
        If it is ``None``, then it can receive inputs of any type.

    """

    domain_type = None
    target_type = None

    @abstractmethod
    def transform(self, point):
        """Transform a point from domain dimension to the target dimension."""

    @abstractmethod
    def reverse(self, transformed_point, index=None):
        """Reverse transform a point from target dimension to the domain dimension."""

    def infer_target_shape(self, shape):
        """Return the shape of the dimension after transformation."""
        return shape

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return f"{self.__class__.__name__}({what})"

    def _get_hashable_members(self):
        return (self.__class__.__name__, self.domain_type, self.target_type)

    # pylint:disable=protected-access
    def __eq__(self, other):
        """Return True if other is the same transformed dimension as self"""
        if not isinstance(other, Transformer):
            return False
        return self._get_hashable_members() == other._get_hashable_members()


class Identity(Transformer):
    """Implement an identity transformation. Everything as it is."""

    def __init__(self, domain_type=None):
        self._domain_type = domain_type

    @property
    def first(self):
        """Signals to ReshapedSpace whether this dimension should be used for `reverse`"""
        return True

    def transform(self, point):
        """Return `point` as it is."""
        return point

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Return `transformed_point` as it is."""
        if index is not None:
            return transformed_point[index]
        return transformed_point

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return what

    @property
    def domain_type(self):
        """Return declared domain type on initialization."""
        return self._domain_type

    @property
    def target_type(self):
        """Return domain type as this will be the target in a identity transformation."""
        return self.domain_type


class Compose(Transformer):
    """Initialize composite transformer with a list of `Transformer` objects
    and domain type on which it will be applied.
    """

    def __init__(self, transformers, base_domain_type=None):
        try:
            self.apply = transformers[-1]
        except IndexError:
            self.apply = Identity()
        if len(transformers) > 1:
            self.composition = Compose(transformers[:-1], base_domain_type)
        else:
            self.composition = Identity(base_domain_type)
        assert (
            self.apply.domain_type is None
            or self.composition.target_type == self.apply.domain_type
        )

    def transform(self, point):
        """Apply transformers in the increasing order of the `transformers` list."""
        point = self.composition.transform(point)
        return self.apply.transform(point)

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Reverse transformation by reversing in the opposite order of the `transformers` list."""
        transformed_point = self.apply.reverse(transformed_point)
        return self.composition.reverse(transformed_point)

    def interval(self, alpha=1.0):
        """Return interval of composed transformation."""
        if hasattr(self.apply, "interval"):
            return self.apply.interval(alpha)

        return None

    def infer_target_shape(self, shape):
        """Return the shape of the dimension after transformation."""
        shape = self.composition.infer_target_shape(shape)
        return self.apply.infer_target_shape(shape)

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return self.apply.repr_format(self.composition.repr_format(what))

    @property
    def domain_type(self):
        """Return base domain type."""
        return self.composition.domain_type

    @property
    def target_type(self):
        """Infer type of the transformation target."""
        type_before = self.composition.target_type
        type_after = self.apply.target_type
        return type_after if type_after else type_before

    # pylint:disable=protected-access
    def _get_hashable_members(self):
        return (
            (self.__class__.__name__,)
            + self.apply._get_hashable_members()
            + self.composition._get_hashable_members()
        )


class Reverse(Transformer):
    """Apply the reverse transformation that another one would do."""

    def __init__(self, transformer: Transformer):
        assert not isinstance(
            transformer, OneHotEncode
        ), "real to categorical is pointless"
        self.transformer = transformer

    def transform(self, point):
        """Use `reserve` of composed `transformer`."""
        return self.transformer.reverse(point)

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Use `transform` of composed `transformer`."""
        return self.transformer.transform(transformed_point)

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return f"{self.__class__.__name__}{self.transformer.repr_format(what)}"

    @property
    def target_type(self):
        """Return `domain_type` of composed `transformer`."""
        return self.transformer.domain_type

    @property
    def domain_type(self):
        """Return `target_type` of composed `transformer`."""
        return self.transformer.target_type


class Precision(Transformer):
    """Round real numbers to requested precision."""

    domain_type = "real"
    target_type = "real"

    def __init__(self, precision=4):
        self.precision = precision

    def transform(self, point):
        """Round `point` to the requested precision, as numpy arrays."""
        # numpy.format_float_scientific precision starts at 0
        if isinstance(point, (list, tuple)) or (
            isinstance(point, numpy.ndarray) and point.shape
        ):
            format_float = numpy.vectorize(
                lambda x: numpy.format_float_scientific(x, precision=self.precision - 1)
            )
            point = format_float(point)
            to_float = numpy.vectorize(float)
            point = to_float(point)
        else:
            point = float(
                numpy.format_float_scientific(point, precision=self.precision - 1)
            )

        return numpy.asarray(point)

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Cast `transformed_point` to floats, as numpy arrays."""
        return self.transform(transformed_point)

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return f"{self.__class__.__name__}({self.precision}, {what})"


class Quantize(Transformer):
    """Transform real numbers to integers, violating injection."""

    domain_type = "real"
    target_type = "integer"

    def transform(self, point):
        """Round `point` and then cast to integers, as numpy arrays."""
        quantized = numpy.round(numpy.asarray(point)).astype(int)

        if numpy.any(numpy.isinf(point)):
            isinf = int(numpy.isinf(point))
            quantized = (
                isinf * (quantized - 1) * int(numpy.sign(point))
                + (1 - isinf) * (quantized - 1)
            ).astype(int)

        return quantized

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Cast `transformed_point` to floats, as numpy arrays."""
        return numpy.asarray(transformed_point).astype(float)


class Enumerate(Transformer):
    """Enumerate categories.

    Effectively transform from a list of objects to a range of integers.
    """

    domain_type = "categorical"
    target_type = "integer"

    def __init__(self, categories):
        self.categories = categories
        map_dict = {cat: i for i, cat in enumerate(categories)}
        self._map = numpy.vectorize(lambda x: map_dict[x], otypes="i")
        self._imap = numpy.vectorize(lambda x: categories[x], otypes=[numpy.object])

    def __deepcopy__(self, memo):
        """Make a deepcopy"""
        return type(self)(self.categories)

    def transform(self, point):
        """Return integers corresponding uniquely to the categories in `point`."""
        return self._map(point)

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Return categories corresponding to their positions inside `transformed_point`."""
        return self._imap(transformed_point)

    # pylint:disable=unused-argument
    def interval(self, alpha=1.0):
        """Return the interval for the enumerated choices."""
        return (0, len(self.categories) - 1)


class OneHotEncode(Transformer):
    """Encode categories to a 1-hot integer space representation."""

    domain_type = "integer"
    target_type = "real"

    def __init__(self, bound: int):
        self.num_cats = bound

    def transform(self, point):
        """Match a `point` containing integers to real vector representations of them.

        If the upper bound of integers supported by an instance of `OneHotEncode`
        is less or equal to 2, then cast them to floats.

        .. note:: This transformation possibly appends one more tensor dimension to `point`.
        """
        point_ = numpy.asarray(point)
        assert (
            numpy.all(point_ < self.num_cats)
            and numpy.all(point_ >= 0)
            and numpy.all(point_ % 1 == 0)
        )

        if self.num_cats <= 2:
            return numpy.asarray(point_, dtype=float)

        hot = numpy.zeros(self.infer_target_shape(point_.shape))
        grid = numpy.meshgrid(
            *[numpy.arange(dim) for dim in point_.shape], indexing="ij"
        )
        hot[tuple(grid + [point_])] = 1
        return hot

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Match real vector representations to integers using an argmax function.

        If the number of dimensions is exactly 2, then use 0.5 as a decision boundary,
        and convert representation to integers 0 or 1.

        If the number of dimensions is exactly 1, then return zeros.

        .. note:: This reverse transformation possibly removes the last tensor dimension
           from `transformed_point`.
        """

        point_ = numpy.asarray(transformed_point)
        if self.num_cats == 2:
            return (point_ > 0.5).astype(int)
        elif self.num_cats == 1:
            return numpy.zeros_like(point_, dtype=int)

        assert point_.shape[-1] == self.num_cats
        return point_.argmax(axis=-1)

    # pylint:disable=unused-argument
    def interval(self, alpha=1.0):
        """Return the interval for the one-hot encoding in proper shape."""
        if self.num_cats == 2:
            return 0, 1
        else:
            low = numpy.zeros(self.num_cats)
            high = numpy.ones(self.num_cats)

            return low, high

    def infer_target_shape(self, shape):
        """Infer that transformed points will have one more tensor dimension,
        if the number of supported integers to transform is larger than 2.
        """
        return tuple(list(shape) + [self.num_cats]) if self.num_cats > 2 else shape

    def _get_hashable_members(self):
        return super()._get_hashable_members() + (self.num_cats,)


class Linearize(Transformer):
    """Transform real numbers from loguniform to linear."""

    domain_type = "real"
    target_type = "real"

    def transform(self, point):
        """Linearize logarithmic distribution."""
        return numpy.log(numpy.asarray(point))

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Turn linear distribution to logarithmic distribution."""
        return numpy.exp(numpy.asarray(transformed_point))


class View(Transformer):
    """Look-up single index in a dimensions with shape > 1"""

    def __init__(self, shape, index, domain_type=None):
        self.shape = shape
        self.index = index
        self._domain_type = domain_type

    @property
    def first(self):
        """Signals to ReshapedSpace whether this dimension should be used for `reverse`"""
        return sum(self.index) == 0

    def transform(self, point):
        """Only return one element of the group"""
        return numpy.array(point)[self.index]

    def reverse(self, transformed_point, index=None):
        """Only return packend point if view of first element, otherwise drop."""
        subset = transformed_point[index : index + numpy.prod(self.shape)]
        return numpy.array(subset).reshape(self.shape)

    def interval(self, interval):
        """Return corresponding view from interval"""
        return (interval[0][self.index], interval[1][self.index])

    @property
    def domain_type(self):
        """Return declared domain type on initialization."""
        return self._domain_type

    @property
    def target_type(self):
        """Return domain type as this will be the target in flatten transformation."""
        return self.domain_type

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return (
            f"{self.__class__.__name__}(shape={self.shape}, index={self.index}, {what})"
        )


class TransformedDimension:
    """Duck-type :class:`orion.algo.space.Dimension` to mimic its functionality,
    while transform automatically and appropriately an underlying
    :class:`orion.algo.space.Dimension` object according to a `Transformer` object.
    """

    NO_DEFAULT_VALUE = Dimension.NO_DEFAULT_VALUE

    def __init__(self, transformer, original_dimension):
        self.original_dimension = original_dimension
        self.transformer = transformer

    def transform(self, point):
        """Expose `Transformer.transform` interface from underlying instance."""
        return self.transformer.transform(point)

    # pylint:disable=unused-argument
    def reverse(self, transformed_point, index=None):
        """Expose `Transformer.reverse` interface from underlying instance."""
        return self.transformer.reverse(transformed_point)

    def interval(self, alpha=1.0):
        """Map the interval bounds to the transformed ones."""
        if hasattr(self.transformer, "interval"):
            interval = self.transformer.interval()
            if interval:
                return interval
        if self.original_dimension.type == "categorical":
            return self.original_dimension.categories

        low, high = self.original_dimension.interval(alpha)

        return self.transform(low), self.transform(high)

    def __contains__(self, point):
        """Reverse transform and ask the original dimension if it is a possible
        sample.
        """
        try:
            orig_point = self.reverse(point)
        except AssertionError:
            return False
        return orig_point in self.original_dimension

    def __repr__(self):
        """Represent the object as a string."""
        return self.transformer.repr_format(repr(self.original_dimension))

    # pylint:disable=protected-access
    def __eq__(self, other):
        """Return True if other is the same transformed dimension as self"""
        if not (hasattr(other, "transformer") and hasattr(other, "original_dimension")):
            return False

        return (
            self.transformer == other.transformer
            and self.original_dimension == other.original_dimension
        )

    def __hash__(self):
        """Hash of the transformed dimension"""
        return hash(self._get_hashable_members())

    # pylint:disable=protected-access
    def _get_hashable_members(self):
        """Hashable members of transformation and original dimension"""
        return (
            self.transformer._get_hashable_members()
            + self.original_dimension._get_hashable_members()
        )

    def validate(self):
        """Validate original_dimension"""
        self.original_dimension.validate()

    @property
    def name(self):
        """Do not change the name of the original dimension."""
        return self.original_dimension.name

    @property
    def type(self):
        """Ask transformer which is its target class."""
        type_ = self.transformer.target_type
        return type_ if type_ != "invariant" else self.original_dimension.type

    @property
    def prior_name(self):
        """Do not change the prior name of the original dimension."""
        return self.original_dimension.prior_name

    @property
    def shape(self):
        """Wrap original shape with transformer, because it may have changed."""
        return self.transformer.infer_target_shape(self.original_dimension.shape)

    @property
    def cardinality(self):
        """Wrap original :class:`orion.algo.space.Dimension` capacity"""
        # May be a discretized real, must reduce cardinality
        if self.type == "integer":
            return Integer.get_cardinality(self.shape, self.interval())

        # Else we don't care what transformation is.
        return self.original_dimension.cardinality

    @property
    def default_value(self):
        """Return the default value for this dimensions"""
        if (
            self.original_dimension.default_value
            is self.original_dimension.NO_DEFAULT_VALUE
        ):
            return self.NO_DEFAULT_VALUE

        return self.transform(self.original_dimension.default_value)


class ReshapedDimension(TransformedDimension):
    """Duck-type :class:`orion.algo.space.Dimension` to mimic its functionality."""

    def __init__(self, transformer, original_dimension, index, name=None):
        super().__init__(transformer, original_dimension)
        if name is None:
            name = original_dimension.name
        self._name = name
        self.index = index

    @property
    def first(self):
        """Signals to ReshapedSpace whether this dimension should be used for `reverse`"""
        return self.transformer.first

    def transform(self, point):
        """Expose `Transformer.transform` interface from underlying instance."""
        return self.transformer.transform(point)

    def reverse(self, transformed_point, index=None):
        """Expose `Transformer.reverse` interface from underlying instance."""
        return self.transformer.reverse(transformed_point, index)

    def interval(self, alpha=1.0):
        """Map the interval bounds to the transformed ones."""
        interval = self.original_dimension.interval(alpha)
        if hasattr(interval[0], "shape") and numpy.prod(interval[0].shape) > 1:
            return self.transformer.interval(interval)

        return interval

    @property
    def cardinality(self):
        """Compute cardinality"""
        cardinality = super().cardinality
        if isinstance(self.transformer, View):
            cardinality /= numpy.prod(self.transformer.shape)

        return cardinality

    def cast(self, point):
        """Cast a point according to original_dimension and then transform it"""
        return self.original_dimension.cast(point)

    @property
    def shape(self):
        """Shape is fixed to ()."""
        return ()

    @property
    def name(self):
        """Name of the view"""
        return self._name


class TransformedSpace(Space):
    """Wrap the :class:`orion.algo.space.Space` to support transformation methods.

    Parameters
    ----------
    space: `orion.algo.space.Space`
       Original problem's definition of parameter space.

    """

    contains = TransformedDimension

    def __init__(self, space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_space = space

    def transform(self, trial):
        """Transform a point that was in the original space to be in this one."""
        transformed_point = tuple(
            dim.transform(flatten(trial.params)[name]) for name, dim in self.items()
        )

        return change_trial_params(trial, transformed_point, self)

    def reverse(self, transformed_trial):
        """Reverses transformation so that a point from this `TransformedSpace`
        to be in the original one.
        """
        reversed_point = tuple(
            dim.reverse(flatten(transformed_trial.params)[name])
            for name, dim in self.items()
        )

        return change_trial_params(
            transformed_trial,
            reversed_point,
            self,
        )

    def sample(self, n_samples=1, seed=None):
        """Sample from the original dimension and forward transform them."""
        trials = self._original_space.sample(n_samples=n_samples, seed=seed)
        return [self.transform(trial) for trial in trials]


class ReshapedSpace(Space):
    """Wrap the `TransformedSpace` to support reshape methods.

    Parameters
    ----------
    space: `orion.core.worker.TransformedSpace`
       Transformed version of the orinigal problem's definition of parameter space.

    """

    contains = ReshapedDimension

    def __init__(self, original_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_space = original_space

    @property
    def original(self):
        """Original space without reshape or transformations"""
        return self._original_space

    def transform(self, trial):
        """Transform a point that was in the original space to be in this one."""
        return self.reshape(self.original.transform(trial))

    def reverse(self, transformed_trial):
        """Reverses transformation so that a point from this `ReshapedSpace` to be in the original
        one.
        """
        return self.original.reverse(self.restore_shape(transformed_trial))

    def reshape(self, trial):
        """Reshape the point"""
        point = format_trials.trial_to_tuple(trial, self._original_space)
        reshaped_point = []
        for dim in self.values():
            reshaped_point.append(dim.transform(point[dim.index]))

        return change_trial_params(trial, reshaped_point, self)

    def restore_shape(self, transformed_trial):
        """Restore shape."""
        transformed_point = format_trials.trial_to_tuple(transformed_trial, self)
        original_keys = self._original_space.keys()
        point = [None for _ in original_keys]
        for index, dim in enumerate(self.values()):
            if dim.first:
                point_index = original_keys.index(dim.original_dimension.name)
                point[point_index] = dim.reverse(transformed_point, index)

        return change_trial_params(transformed_trial, point, self._original_space)

    def sample(self, n_samples=1, seed=None):
        """Sample from the original dimension and forward transform them."""
        trials = self.original.sample(n_samples=n_samples, seed=seed)
        return [self.reshape(trial) for trial in trials]

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

        return self.restore_shape(key_or_trial) in self.original

    @property
    def cardinality(self):
        """Reshape does not affect cardinality"""
        return self.original.cardinality


def change_trial_params(trial, point, space):
    """Convert params in Param objects and update trial"""
    new_trial = copy.copy(trial)
    # pylint: disable=protected-access
    new_trial._params = format_trials.tuple_to_trial(point, space)._params
    return new_trial
