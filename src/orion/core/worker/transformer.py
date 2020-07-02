# -*- coding: utf-8 -*-
# flake8: noqa: D102
# pylint: disable=no-self-use
"""
:mod:`orion.core.worker.transformer` -- Perform transformations on Dimensions
=============================================================================

.. module:: transformer
   :platform: Unix
   :synopsis: Provide functions and classes to build a Space which an
      algorithm can operate on.

"""
from abc import (ABCMeta, abstractmethod)

import numpy

from orion.algo.space import (Dimension, Space)


# pylint: disable=too-many-branches
def build_required_space(requirements, original_space):
    """Build a `Space` object which agrees to the `requirements` imposed
    by the desired optimization algorithm.

    It uses appropriate cascade of `Transformer` objects per `Dimension`
    contained in `original_space`.

    Parameters
    ----------
    requirements : list of str or str
       Describes requirements that an algorithm needs a parameter space to have
       in order to be able to operate on it. In case it is a list, the infered
       transformations are going to be applied from the first item in the list to
       the last.
    original_space : `orion.algo.space.Space`
       Original problem's definition of parameter space given by the user to Oríon.

    Supported Requirements
    ----------------------
     * Null requirement; use problem's parameter space as it is defined
     * ``'real'``: transform every dimension to a `orion.algo.space.Real` one
     * ``'integer'``: transform every dimension to a `orion.algo.space.Integer` one

    """
    requirements = requirements if isinstance(requirements, list) else [requirements]
    if not requirements:
        requirements = [None]

    space = TransformedSpace()
    for dim in original_space.values():
        transformers = []
        type_ = dim.type
        base_domain_type = type_
        for requirement in requirements:
            if type_ == 'real' and requirement in ('real', None) and dim.precision is not None:
                transformers.append(Precision(dim.precision))
            elif type_ == 'real' and requirement in ('real', None):
                pass
            elif type_ == 'real' and requirement == 'integer':
                transformers.append(Quantize())
            elif type_ == 'integer' and requirement in ('integer', None):
                pass
            elif type_ == 'integer' and requirement == 'real':
                transformers.append(Reverse(Quantize()))
            elif type_ == 'categorical' and requirement == 'real':
                transformers.extend([Enumerate(dim.categories),
                                     OneHotEncode(len(dim.categories))])
            elif type_ == 'categorical' and requirement == 'integer':
                transformers.append(Enumerate(dim.categories))
            elif type_ == 'categorical' and requirement is None:
                pass
            elif type_ == 'fidelity' and requirement is None:
                pass
            else:
                raise TypeError("Unsupported dimension type ('{}') "
                                "or requirement ('{}')".format(type_, requirement))
            try:
                last_type = transformers[-1].target_type
                type_ = last_type if last_type != 'invariant' else type_
            except IndexError:
                pass
        space.register(TransformedDimension(Compose(transformers, base_domain_type),
                                            dim))
    return space


class Transformer(object, metaclass=ABCMeta):
    """Define an (injective) function and its inverse. Base transformation class.

    :attr:`target_type` defines the type of the target space of the forward function.
    It can provide one of the values: ``['real', 'integer', 'categorical']``.

    :attr:`domain_type` is similar to `target_type` but it refers to the domain.
    If it is ``None``, then it can receive inputs of any type.
    """

    domain_type = None
    target_type = None

    @abstractmethod
    def transform(self, point):
        """Transform a point from domain dimension to the target dimension."""
        pass

    @abstractmethod
    def reverse(self, transformed_point):
        """Reverse transform a point from target dimension to the domain dimension."""
        pass

    def infer_target_shape(self, shape):
        """Return the shape of the dimension after transformation."""
        return shape

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return "{}({})".format(self.__class__.__name__, what)

    def _get_hashable_members(self):
        print((self.__class__.__name__, self.domain_type, self.target_type))
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

    def transform(self, point):
        """Return `point` as it is."""
        return point

    def reverse(self, transformed_point):
        """Return `transformed_point` as it is."""
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
    and domain type on which it will be applied."""

    def __init__(self, transformers, base_domain_type=None):
        try:
            self.apply = transformers.pop()
        except IndexError:
            self.apply = Identity()
        if transformers:
            self.composition = Compose(transformers, base_domain_type)
        else:
            self.composition = Identity(base_domain_type)
        assert self.apply.domain_type is None or \
            self.composition.target_type == self.apply.domain_type

    def transform(self, point):
        """Apply transformers in the increasing order of the `transformers` list."""
        point = self.composition.transform(point)
        return self.apply.transform(point)

    def reverse(self, transformed_point):
        """Reverse transformation by reversing in the opposite order of the `transformers` list."""
        transformed_point = self.apply.reverse(transformed_point)
        return self.composition.reverse(transformed_point)

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
        """Infer type of the tranformation target."""
        type_before = self.composition.target_type
        type_after = self.apply.target_type
        return type_after if type_after else type_before

    # pylint:disable=protected-access
    def _get_hashable_members(self):
        return ((self.__class__.__name__, ) +
                self.apply._get_hashable_members() +
                self.composition._get_hashable_members())


class Reverse(Transformer):
    """Apply the reverse transformation that another one would do."""

    def __init__(self, transformer: Transformer):
        assert not isinstance(transformer, OneHotEncode), "real to categorical is pointless"
        self.transformer = transformer

    def transform(self, point):
        """Use `reserve` of composed `transformer`."""
        return self.transformer.reverse(point)

    def reverse(self, transformed_point):
        """Use `transform` of composed `transformer`."""
        return self.transformer.transform(transformed_point)

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return "{}{}".format(self.__class__.__name__, self.transformer.repr_format(what))

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

    domain_type = 'real'
    target_type = 'real'

    def __init__(self, precision=4):
        self.precision = precision

    def transform(self, point):
        """Round `point` to the requested precision, as numpy arrays."""
        # numpy.format_float_scientific precision starts at 0
        if isinstance(point, (list, tuple)) or (isinstance(point, numpy.ndarray) and point.shape):
            point = map(lambda x: numpy.format_float_scientific(x, precision=self.precision - 1),
                        point)
            point = list(map(float, point))
        else:
            point = float(numpy.format_float_scientific(point, precision=self.precision - 1))

        return numpy.asarray(point)

    def reverse(self, transformed_point):
        """Cast `transformed_point` to floats, as numpy arrays."""
        return self.transform(transformed_point)

    def repr_format(self, what):
        """Format a string for calling ``__repr__`` in `TransformedDimension`."""
        return "{}({}, {})".format(self.__class__.__name__, self.precision, what)


class Quantize(Transformer):
    """Transform real numbers to integers, violating injection."""

    domain_type = 'real'
    target_type = 'integer'

    def transform(self, point):
        """Cast `point` to the floor and then to integers, as numpy arrays."""
        return numpy.floor(numpy.asarray(point)).astype(int)

    def reverse(self, transformed_point):
        """Cast `transformed_point` to floats, as numpy arrays."""
        return numpy.asarray(transformed_point).astype(float)


class Enumerate(Transformer):
    """Enumerate categories.

    Effectively transform from a list of objects to a range of integers.
    """

    domain_type = 'categorical'
    target_type = 'integer'

    def __init__(self, categories):
        self.categories = categories
        map_dict = {cat: i for i, cat in enumerate(categories)}
        self._map = numpy.vectorize(lambda x: map_dict[x], otypes='i')
        self._imap = numpy.vectorize(lambda x: categories[x], otypes=[numpy.object])

    def __deepcopy__(self, memo):
        """Make a deepcopy"""
        return type(self)(self.categories)

    def transform(self, point):
        """Return integers corresponding uniquely to the categories in `point`.

        :rtype: numpy.ndarray, integer
        """
        return self._map(point)

    def reverse(self, transformed_point):
        """Return categories corresponding to their positions inside `transformed_point`.

        :rtype: numpy.ndarray, numpy.object
        """
        return self._imap(transformed_point)


class OneHotEncode(Transformer):
    """Encode categories to a 1-hot integer space representation."""

    domain_type = 'integer'
    target_type = 'real'

    def __init__(self, bound: int):
        self.num_cats = bound

    def transform(self, point):
        """Match a `point` containing integers to real vector representations of them.

        If the upper bound of integers supported by an instance of `OneHotEncode`
        is less or equal to 2, then cast them to floats.

        .. note:: This transformation possibly appends one more tensor dimension to `point`.
        """
        point_ = numpy.asarray(point)
        assert numpy.all(point_ < self.num_cats) and numpy.all(point_ >= 0) and\
            numpy.all(point_ % 1 == 0)

        if self.num_cats <= 2:
            return numpy.asarray(point_, dtype=float)

        hot = numpy.zeros(self.infer_target_shape(point_.shape))
        grid = numpy.meshgrid(*[numpy.arange(dim) for dim in point_.shape],
                              indexing='ij')
        hot[grid + [point_]] = 1
        return hot

    def reverse(self, transformed_point):
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

    def infer_target_shape(self, shape):
        """Infer that transformed points will have one more tensor dimension,
        if the number of supported integers to transform is larger than 2.
        """
        return tuple(list(shape) + [self.num_cats]) if self.num_cats > 2 else shape

    def _get_hashable_members(self):
        return super(OneHotEncode, self)._get_hashable_members() + (self.num_cats, )


# pylint:disable=too-many-public-methods
class TransformedDimension(object):
    """Duck-type `Dimension` to mimic its functionality,
    while transform automatically and appropriately an underlying `Dimension` object
    according to a `Transformer` object.
    """

    NO_DEFAULT_VALUE = Dimension.NO_DEFAULT_VALUE

    def __init__(self, transformer: Transformer, original_dimension: Dimension):
        """Initialize a `TransformedDimension` with an `original_dimension` object
        and the `transformer` that will be used.
        """
        self.original_dimension = original_dimension
        self.transformer = transformer

    def transform(self, point):
        """Expose `Transformer.transform` interface from underlying instance."""
        return self.transformer.transform(point)

    def reverse(self, transformed_point):
        """Expose `Transformer.reverse` interface from underlying instance."""
        return self.transformer.reverse(transformed_point)

    def sample(self, n_samples=1, seed=None):
        """Sample from the original dimension and forward transform them."""
        samples = self.original_dimension.sample(n_samples, seed)
        return [self.transform(sample) for sample in samples]

    def interval(self, alpha=1.0):
        """Map the interval bounds to the transformed ones."""
        if self.original_dimension.prior_name == 'choices':
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

        return (self.transformer == other.transformer and
                self.original_dimension == other.original_dimension)

    def __hash__(self):
        return hash(self._get_hashable_members())

    # pylint:disable=protected-access
    def _get_hashable_members(self):
        print(self.transformer._get_hashable_members())
        print(self.original_dimension._get_hashable_members())
        return (self.transformer._get_hashable_members() +
                self.original_dimension._get_hashable_members())

    def validate(self):
        """Validate original_dimension"""
        self.original_dimension.validate()

    def get_prior_string(self):
        """Do not change the prior string of original dimension."""
        return self.transformer.repr_format(self.original_dimension.get_prior_string())

    def get_string(self):
        """Do not change the string of original dimension."""
        original_prior = self.original_dimension.get_prior_string()
        original_string = self.original_dimension.get_string()
        return original_string.replace(original_prior, self.get_prior_string())

    @property
    def name(self):
        """Do not change the name of the original dimension."""
        return self.original_dimension.name

    @property
    def type(self):
        """Ask transformer which is its target class."""
        type_ = self.transformer.target_type
        return type_ if type_ != 'invariant' else self.original_dimension.type

    @property
    def prior_name(self):
        """Do not change the prior name of the original dimension."""
        return self.original_dimension.prior_name

    @property
    def shape(self):
        """Wrap original shape with transformer, because it may have changed."""
        return self.transformer.infer_target_shape(self.original_dimension.shape)

    @property
    def default_value(self):
        """Wrap original default value."""
        defval = self.original_dimension.default_value
        return self.transform(defval) if defval is not None else None

    def cast(self, point):
        """Cast a point according to original_dimension and then transform it"""
        return self.transform(self.original_dimension.cast(point))

    @property
    def cardinality(self):
        """Wrap original `Dimension` capacity"""
        return self.original_dimension.cardinality


class TransformedSpace(Space):
    """Wrap the `Space` to support transformation methods."""

    contains = TransformedDimension

    def transform(self, point):
        """Transform a point that was in the original space to be in this one."""
        return tuple([dim.transform(point[i]) for i, dim in enumerate(self.values())])

    def reverse(self, transformed_point):
        """Reverses transformation so that a point from this `TransformedSpace`
        to be in the original one.
        """
        return tuple([dim.reverse(transformed_point[i]) for i, dim in enumerate(self.values())])
