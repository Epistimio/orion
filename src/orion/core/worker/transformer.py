# -*- coding: utf-8 -*-
# flake8: noqa: D102
# pylint: disable=missing-docstring,no-self-use
"""
:mod:`orion.core.worker.transformer` -- Perform operations on Dimensions
========================================================================

.. module:: transformer
   :platform: Unix
   :synopsis: Provide functions and classes to build a Space which an
      algorithm can operate on.

"""
from abc import (ABCMeta, abstractmethod)

import numpy

from orion.algo.space import Space


def build_required_space(requirements, original_space):
    """Build a `Space` object which agrees to the `requirements` imposed
    by the desired optimization algorithm.

    It uses appropriate cascade of `Transformer` objects per `Dimension`
    contained in `original_space`.
    """
    requirements = requirements if isinstance(requirements, list) else [requirements]
    space = TransformedSpace()
    for dim in original_space.values():
        transformers = []
        type_ = dim.type
        for requirement in requirements:
            # TODO Write classes and instantiate transformers accordingly
            if type_ == 'real':
                pass
            elif type_ == 'integer':
                pass
            elif type_ == 'categorical':
                pass
            else:
                raise TypeError("Unsupported dimension type '{}'".format(type_))
            last_type = transformers[-1].target_type
            type_ = last_type if last_type != 'invariant' else type_
        space.register(TransformedDimension(Compose(transformers), dim))
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


class Identity(Transformer):
    """Implement an identity transformation. Everything as it is."""

    def __init__(self, domain_type=None):
        self._domain_type = domain_type

    def transform(self, point):
        return point

    def reverse(self, transformed_point):
        return transformed_point

    def repr_format(self, what):
        return what

    @property
    def domain_type(self):
        return self._domain_type

    @property
    def target_type(self):
        return self.domain_type


class Compose(Transformer):

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
        point = self.composition.transform(point)
        return self.apply.transform(point)

    def reverse(self, transformed_point):
        transformed_point = self.apply.reverse(transformed_point)
        return self.composition.reverse(transformed_point)

    def infer_target_shape(self, shape):
        shape = self.composition.infer_target_shape(shape)
        return self.apply.infer_target_shape(shape)

    def repr_format(self, what):
        return self.apply.repr_format(self.composition.repr_format(what))

    @property
    def domain_type(self):
        return self.composition.domain_type

    @property
    def target_type(self):
        type_before = self.composition.target_type
        type_after = self.apply.target_type
        return type_after if type_after else type_before


class Reverse(Transformer):
    """Apply the reverse transformation that another one would do."""

    def __init__(self, transformer: Transformer):
        self.transformer = transformer

    def transform(self, point):
        return self.transformer.reverse(point)

    def reverse(self, transformed_point):
        return self.transformer.transform(transformed_point)

    def repr_format(self, what):
        return "{}{}".format(self.__class__.__name__, what)

    @property
    def target_type(self):
        return self.transformer.domain_type

    @property
    def domain_type(self):
        return self.transformer.target_type


class Quantize(Transformer):
    """Transform real numbers to integers, violating injection."""

    domain_type = 'real'
    target_type = 'integer'

    def transform(self, point):
        return numpy.asarray(point).round().astype(int)

    def reverse(self, transformed_point):
        return numpy.asarray(transformed_point).astype(float)


class Enumerate(Transformer):
    """Enumerate categories."""

    domain_type = 'categorical'
    target_type = 'integer'

    def __init__(self, categories):
        self.categories = categories
        map_dict = {cat: i for i, cat in enumerate(categories)}
        self._map = numpy.vectorize(lambda x: map_dict[x], otypes='i')
        self._imap = numpy.vectorize(lambda x: categories[x], otypes=[numpy.object])

    def transform(self, point):
        return self._map(point).tolist()

    def reverse(self, transformed_point):
        return self._imap(transformed_point).tolist()


class OneHotEncode(Transformer):
    """Encode categories to a 1-hot integer space representation."""

    domain_type = 'categorical'
    target_type = 'real'

    def __init__(self, categories):
        self.categories = categories
        self.enumerator = Enumerate(categories)
        self.num_cats = len(categories)

    def transform(self, point):
        indices = self.enumerator.transform(point)

        if self.num_cats == 2:
            return indices

        hot = numpy.zeros(self.infer_target_shape(indices.shape))
        grid = numpy.meshgrid(*[numpy.arange(dim) for dim in indices.shape],
                              indexing='ij')
        hot[grid + [indices]] = 1
        return hot

    def reverse(self, transformed_point):
        point_ = numpy.asarray(transformed_point)
        if self.num_cats == 2:
            return self.enumerator.reverse((point_ > 0.5).astype(int))
        return self.enumerator.reverse(point_.argmax(axis=-1))

    def infer_target_shape(self, shape):
        return tuple(list(shape) + [self.num_cats])


class TransformedDimension(object):

    def __init__(self, transformer, original_dimension):
        self.original_dimension = original_dimension
        self.transformer = transformer

    def transform(self, point):
        return self.transformer.transform(point)

    def reverse(self, transformed_point):
        return self.transformer.reverse(transformed_point)

    def sample(self, n_samples=1, seed=None):
        """Sample from the original dimension and forward transform them."""
        samples = self.original_dimension.sample(n_samples, seed)
        return [self.transform(sample) for sample in samples]

    def interval(self, alpha=1.0):
        """Map the interval bounds to the transformed ones."""
        low, high = self.original_dimension.interval(alpha)
        return self.transform(low), self.transform(high)

    def __contains__(self, point):
        """Reverse transform and ask the original dimension if it is a possible
        sample.
        """
        orig_point = self.reverse(point)
        return orig_point in self.original_dimension

    def __repr__(self):
        """Represent the object as a string."""
        return self.transformer.repr_format(repr(self.original_dimension))

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
    def shape(self):
        """Wrap original shape with transformer, because it may have changed."""
        return self.transformer.infer_target_shape(self.original_dimension.shape)


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
