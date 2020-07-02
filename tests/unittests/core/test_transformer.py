#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.transformer`."""
from collections import OrderedDict
import copy

import numpy
import pytest

from orion.algo.space import (Categorical, Dimension, Integer, Real, Space,)
from orion.core.worker.transformer import (build_required_space,
                                           Compose, Enumerate, Identity,
                                           OneHotEncode, Precision, Quantize, Reverse,
                                           TransformedDimension, TransformedSpace,)


class TestIdentity(object):
    """Test subclasses of `Identity` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Identity()
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Identity()
        assert t.domain_type is None
        assert t.target_type is None

        t = Identity('mpogias')
        assert t.domain_type == 'mpogias'
        assert t.target_type == 'mpogias'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Identity()
        assert t.transform('yo') == 'yo'

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Identity()
        assert t.reverse('yo') == 'yo'

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Identity()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Identity()
        assert t.repr_format('asfa') == 'asfa'


class TestReverse(object):
    """Test subclasses of `Reverse` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Reverse(Quantize())
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Reverse(Quantize())
        assert t.domain_type == 'integer'
        assert t.target_type == 'real'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Reverse(Quantize())
        assert t.transform(9) == 9.
        assert t.transform(5) == 5.
        assert numpy.all(t.transform([9, 5]) == numpy.array([9., 5.], dtype=float))

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Reverse(Quantize())
        assert t.reverse(8.6) == 8
        assert t.reverse(5.3) == 5
        assert numpy.all(t.reverse([8.6, 5.3]) == numpy.array([8, 5], dtype=int))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Reverse(Quantize())
        assert t.infer_target_shape((5,)) == (5,)

    def test_no_reverse_one_hot_encode(self):
        """Do NOT support real to categorical."""
        with pytest.raises(AssertionError):
            Reverse(OneHotEncode([1, 2, 3]))

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Reverse(Quantize())
        assert t.repr_format('asfa') == 'ReverseQuantize(asfa)'


class TestCompose(object):
    """Test subclasses of `Compose` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Compose([])
        assert t.domain_type is None
        assert t.target_type is None

        t = Compose([], 'real')
        assert t.domain_type == 'real'
        assert t.target_type == 'real'

        t = Compose([Quantize()], 'real')
        assert t.domain_type == 'real'
        assert t.target_type == 'integer'

        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        assert t.domain_type == 'categorical'
        assert t.target_type == 'real'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        assert numpy.all(t.transform(2) == numpy.array((1., 0., 0.)))
        assert numpy.all(t.transform('asfa') == numpy.array((0., 1., 0.)))
        assert numpy.all(t.transform('ipsi') == numpy.array((0., 0., 1.)))
        with pytest.raises(KeyError):
            t.transform('aafdasfa')
        assert(numpy.all(t.transform([['ipsi', 'asfa'], [2, 'ipsi']]) ==
               numpy.array([[(0., 0., 1.), (0., 1., 0.)], [(1., 0., 0.), (0., 0., 1.)]])))

        t = Compose([Enumerate([2, 'asfa']), OneHotEncode(2)], 'categorical')
        assert t.transform(2) == 0.
        assert t.transform('asfa') == 1.
        with pytest.raises(KeyError):
            t.transform('ipsi')
        assert(numpy.all(t.transform([['asfa', 'asfa'], [2, 'asfa']]) ==
               numpy.array([[1., 1.], [0., 1.]])))

        # for the crazy enough
        t = Compose([Enumerate([2]), OneHotEncode(1)], 'categorical')
        assert t.transform(2) == 0.
        with pytest.raises(KeyError):
            t.transform('ipsi')
        assert numpy.all(t.transform([[2, 2], [2, 2]]) == [[0, 0], [0, 0]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        assert t.reverse((0.9, 0.8, 0.3)) == 2
        assert t.reverse((-0.3, 2., 0.)) == 'asfa'
        assert t.reverse((0., 0., 1.)) == 'ipsi'
        with pytest.raises(AssertionError):
            t.reverse((0., 0., 0., 1.))
        assert(numpy.all(t.reverse(numpy.array([[(0., 0., 1.), (0., 1., 0.)],
                                                [(1., 0., 0.), (0., 0., 1.)]])) ==
               numpy.array([['ipsi', 'asfa'], [2, 'ipsi']], dtype=numpy.object)))

        t = Compose([Enumerate([2, 'asfa']), OneHotEncode(2)], 'categorical')
        assert t.reverse(0.3) == 2
        assert t.reverse(2.) == 'asfa'
        assert numpy.all(t.reverse((0., 0., 0., 1.)) == numpy.array([2, 2, 2, 'asfa'],
                                                                    dtype=numpy.object))
        assert(numpy.all(t.reverse(numpy.array([[0.55, 3.], [-0.6, 1.]])) ==
               numpy.array([['asfa', 'asfa'], [2, 'asfa']], dtype=numpy.object)))

        # for the crazy enough
        t = Compose([Enumerate([2]), OneHotEncode(1)], 'categorical')
        assert t.reverse(0) == 2
        assert t.reverse(5.0) == 2
        assert t.reverse(0.2) == 2
        assert t.reverse(-0.2) == 2
        assert numpy.all(t.reverse([[0.5, 0], [1.0, 55]]) == numpy.array([[2, 2], [2, 2]],
                                                                         dtype=numpy.object))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        assert t.infer_target_shape((2, 5)) == (2, 5, 3)

        t = Compose([Enumerate([2, 'asfa']), OneHotEncode(2)], 'categorical')
        assert t.infer_target_shape((2, 5)) == (2, 5)

        t = Compose([Enumerate([2]), OneHotEncode(1)], 'categorical')
        assert t.infer_target_shape((2, 5)) == (2, 5)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Compose([Enumerate([2, 'asfa', 'ipsi']), OneHotEncode(3)], 'categorical')
        assert t.repr_format('asfa') == 'OneHotEncode(Enumerate(asfa))'


class TestPrecision(object):
    """Test subclasses of `Precision` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Precision()
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Precision()
        assert t.domain_type == 'real'
        assert t.target_type == 'real'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Precision(precision=4)
        assert t.transform(8.654321098) == 8.654
        assert t.transform(0.000123456789) == 0.0001235
        assert numpy.all(t.transform([8.654321098, 0.000123456789]) ==
                         numpy.array([8.654, 0.0001235], dtype=float))

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Precision()
        assert t.reverse(9.) == 9.
        assert t.reverse(5.) == 5.
        assert numpy.all(t.reverse([9., 5.]) == numpy.array([9., 5.], dtype=float))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Precision()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Precision()
        assert t.repr_format('asfa') == 'Precision(4, asfa)'


class TestQuantize(object):
    """Test subclasses of `Quantize` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Quantize()
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Quantize()
        assert t.domain_type == 'real'
        assert t.target_type == 'integer'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Quantize()
        assert t.transform(8.6) == 8
        assert t.transform(5.3) == 5
        assert numpy.all(t.transform([8.6, 5.3]) == numpy.array([8, 5], dtype=int))

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Quantize()
        assert t.reverse(9) == 9.
        assert t.reverse(5) == 5.
        assert numpy.all(t.reverse([9, 5]) == numpy.array([9., 5.], dtype=float))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Quantize()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Quantize()
        assert t.repr_format('asfa') == 'Quantize(asfa)'


class TestEnumerate(object):
    """Test subclasses of `Enumerate` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Enumerate([2, 'asfa', 'ipsi'])
        # Copy won't fail if vectorized function is not called at least once.
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Enumerate([2, 'asfa', 'ipsi'])
        assert t.domain_type == 'categorical'
        assert t.target_type == 'integer'

    def test_transform(self):
        """Check if it transforms properly."""
        t = Enumerate([2, 'asfa', 'ipsi'])
        assert t.transform(2) == 0
        assert t.transform('asfa') == 1
        assert t.transform('ipsi') == 2
        with pytest.raises(KeyError):
            t.transform('aafdasfa')
        assert numpy.all(t.transform([['ipsi', 'asfa'], [2, 'ipsi']]) == [[2, 1], [0, 2]])

        # for the crazy enough
        t = Enumerate([2])
        assert t.transform(2) == 0
        with pytest.raises(KeyError):
            t.transform('aafdasfa')
        assert numpy.all(t.transform([[2, 2], [2, 2]]) == [[0, 0], [0, 0]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Enumerate([2, 'asfa', 'ipsi'])
        assert t.reverse(0) == 2
        assert t.reverse(1) == 'asfa'
        assert t.reverse(2) == 'ipsi'
        with pytest.raises(IndexError):
            t.reverse(3)
        assert numpy.all(t.reverse([[2, 1], [0, 2]]) == numpy.array([['ipsi', 'asfa'], [2, 'ipsi']],
                                                                    dtype=numpy.object))

        # for the crazy enough
        t = Enumerate([2])
        assert t.reverse(0) == 2
        with pytest.raises(IndexError):
            t.reverse(1)
        assert numpy.all(t.reverse([[0, 0], [0, 0]]) == numpy.array([[2, 2], [2, 2]],
                                                                    dtype=numpy.object))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Enumerate([2, 'asfa', 'ipsi'])
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Enumerate([2, 'asfa', 'ipsi'])
        assert t.repr_format('asfa') == 'Enumerate(asfa)'


class TestOneHotEncode(object):
    """Test subclasses of `OneHotEncode` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = OneHotEncode(3)
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = OneHotEncode(3)
        assert t.domain_type == 'integer'
        assert t.target_type == 'real'

    def test_transform(self):
        """Check if it transforms properly."""
        t = OneHotEncode(3)
        assert numpy.all(t.transform(0) == numpy.array((1., 0., 0.)))
        assert numpy.all(t.transform(1) == numpy.array((0., 1., 0.)))
        assert numpy.all(t.transform(2) == numpy.array((0., 0., 1.)))
        with pytest.raises(AssertionError):
            t.transform(4)
        with pytest.raises(AssertionError):
            t.transform(-1)
        with pytest.raises(AssertionError):
            t.transform(2.2)
        assert(numpy.all(t.transform([[2, 1], [0, 2]]) ==
               numpy.array([[(0., 0., 1.), (0., 1., 0.)], [(1., 0., 0.), (0., 0., 1.)]])))

        t = OneHotEncode(2)
        assert t.transform(0) == 0.
        assert t.transform(1) == 1.
        with pytest.raises(TypeError):
            t.transform('ipsi')
        assert(numpy.all(t.transform([[1, 1], [0, 1]]) ==
               numpy.array([[1., 1.], [0., 1.]])))

        # for the crazy enough
        t = OneHotEncode(1)
        assert t.transform(0) == 0.
        with pytest.raises(TypeError):
            t.transform('ipsi')
        assert numpy.all(t.transform([[0, 0], [0, 0]]) == [[0., 0.], [0., 0.]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = OneHotEncode(3)
        assert t.reverse((0.9, 0.8, 0.3)) == 0
        assert t.reverse((-0.3, 2., 0.)) == 1
        assert t.reverse((0., 0., 1.)) == 2
        with pytest.raises(AssertionError):
            t.reverse((0., 0., 0., 1.))
        assert(numpy.all(t.reverse(numpy.array([[[0., 0., 1.], [0., 1., 0.]],
                                                [[1., 0., 0.], [0., 0., 1.]]])) ==
               numpy.array([[2, 1], [0, 2]], dtype=int)))

        t = OneHotEncode(2)
        assert t.reverse(0.3) == 0
        assert t.reverse(2.) == 1
        assert numpy.all(t.reverse((0., 0., 0., 1.)) == numpy.array([0, 0, 0, 1],
                                                                    dtype=int))
        assert(numpy.all(t.reverse(numpy.array([[0.55, 3.], [-0.6, 1.]])) ==
               numpy.array([[1, 1], [0, 1]], dtype=int)))

        # for the crazy enough
        t = OneHotEncode(1)
        assert t.reverse(0) == 0
        assert t.reverse(5.0) == 0
        assert t.reverse(0.2) == 0
        assert t.reverse(-0.2) == 0
        assert numpy.all(t.reverse([[0.5, 0], [1.0, 55]]) == numpy.array([[0, 0], [0, 0]],
                                                                         dtype=int))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = OneHotEncode(3)
        assert t.infer_target_shape((2, 5)) == (2, 5, 3)

        t = OneHotEncode(2)
        assert t.infer_target_shape((2, 5)) == (2, 5)

        t = OneHotEncode(1)
        assert t.infer_target_shape((2, 5)) == (2, 5)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = OneHotEncode(3)
        assert t.repr_format('asfa') == 'OneHotEncode(asfa)'


@pytest.fixture(scope='module')
def dim():
    """Create an example of `Dimension`."""
    dim = Real('yolo', 'norm', 0.9, shape=(3, 2))
    return dim


@pytest.fixture(scope='module')
def tdim(dim):
    """Create an example of `TransformedDimension`."""
    transformers = [Quantize()]
    tdim = TransformedDimension(Compose(transformers, dim.type),
                                dim)
    return tdim


@pytest.fixture(scope='module')
def dim2():
    """Create a second example of `Dimension`."""
    probs = (0.1, 0.2, 0.3, 0.4)
    categories = ('asdfa', '2', '3', '4')
    categories = OrderedDict(zip(categories, probs))
    dim2 = Categorical('yolo2', categories)
    return dim2


@pytest.fixture(scope='module')
def tdim2(dim2):
    """Create a second example of `TransformedDimension`."""
    transformers = [Enumerate(dim2.categories), OneHotEncode(len(dim2.categories))]
    tdim2 = TransformedDimension(Compose(transformers, dim2.type),
                                 dim2)
    return tdim2


class TestTransformedDimension(object):
    """Check functionality of class `TransformedDimension`."""

    def test_transform(self, tdim):
        """Check method `transform`."""
        assert tdim.transform(8.6) == 8
        assert tdim.transform(5.3) == 5
        assert numpy.all(tdim.transform([8.6, 5.3]) == numpy.array([8, 5], dtype=int))

    def test_reverse(self, tdim):
        """Check method `reverse`."""
        assert tdim.reverse(9) == 9.
        assert tdim.reverse(5) == 5.
        assert numpy.all(tdim.reverse([9, 5]) == numpy.array([9., 5.], dtype=float))

    def test_mimics_Dimension(self, tdim):
        """Mimic `Dimension`.
        Set of `Dimension`'s methods are subset of `TransformedDimension`.
        """
        transformed_dimension_keys = set(TransformedDimension.__dict__.keys())
        # For some reason running all tests have the side-effect of adding an attribute
        # __slotnames__ to TransformedDimension. This attribute is not present when running
        # tests found in test_transformer.py only.
        transformed_dimension_keys.discard('__slotnames__')
        assert ((transformed_dimension_keys ^ set(Dimension.__dict__.keys())) ==
                set(['transform', 'reverse']))

    def test_sample(self, tdim, seed):
        """Check method `sample`."""
        assert numpy.all(tdim.sample(seed=seed) == numpy.array([[1, 0], [3, 0], [1, 2]]))
        samples = tdim.sample(2, seed=seed)
        assert len(samples) == 2
        assert numpy.all(samples[0] == numpy.array([[-1, 0], [1, 0], [-1, 0]]))
        assert numpy.all(samples[1] == numpy.array([[0, 1], [-1, 0], [2, 2]]))

    def test_interval(self, tdim):
        """Check method `interval`."""
        tmp1 = tdim.original_dimension._low
        tmp2 = tdim.original_dimension._high
        tdim.original_dimension._low = -0.6
        tdim.original_dimension._high = 1.2

        assert tdim.interval() == (-1, 1)

        tdim.original_dimension._low = tmp1
        tdim.original_dimension._high = tmp2

    def test_interval_from_categorical(self, tdim2):
        """Check how we should treat interval when original dimension is categorical."""
        assert tdim2.interval() == ('asdfa', '2', '3', '4')

    def test_contains(self, tdim):
        """Check method `__contains__`."""
        assert [[1, 1], [3, 1], [1, 2]] in tdim

        tmp1 = tdim.original_dimension._low
        tmp2 = tdim.original_dimension._high
        tdim.original_dimension._low = -0.6
        tdim.original_dimension._high = 1.2

        assert [[1, 1], [3, 1], [1, 2]] not in tdim

        tdim.original_dimension._low = tmp1
        tdim.original_dimension._high = tmp2

    def test_contains_from_categorical(self, tdim2):
        """Check method `__contains__` when original is categorical."""
        assert (0, 0, 0, 1) in tdim2
        assert (0, 2, 0, 1) in tdim2
        assert (0, 2, 0) not in tdim2

    def test_eq(self, tdim, tdim2):
        """Return True if other is the same transformed dimension as self"""
        assert tdim != tdim2
        assert tdim == copy.deepcopy(tdim)

    def test_hash(self, tdim, tdim2):
        """Test that hash is consistent for identical and different transformed dimensions"""
        assert hash(tdim) != hash(tdim2)
        assert hash(tdim) == hash(copy.deepcopy(tdim))

    def test_get_hashable_members(self, tdim, tdim2):
        """Test that hashable members of the transformed dimensions are the aggregation of
        transformer's and original dimension's hashable members.
        """
        assert (tdim._get_hashable_members() ==
                ('Compose', 'Quantize', 'real', 'integer', 'Identity', 'real', 'real',
                 'yolo', (3, 2), 'real', (0.9,), (), None, 'norm'))
        assert (tdim2._get_hashable_members() ==
                ('Compose', 'OneHotEncode', 'integer', 'real', 4, 'Compose', 'Enumerate',
                 'categorical', 'integer', 'Identity', 'categorical', 'categorical', 'yolo2', (),
                 'categorical', (), (), None, 'Distribution'))

    def test_validate(self, tdim, tdim2):
        """Validate original_dimension"""
        # It pass
        tdim.validate()
        tdim2.validate()

        # We break it
        tdim.original_dimension._kwargs['size'] = (2, )
        tdim2.original_dimension._default_value = 'bad-default'

        # It does not pass
        with pytest.raises(ValueError) as exc:
            tdim.validate()
        assert "Use 'shape' keyword only instead of 'size'." in str(exc.value)

        with pytest.raises(ValueError) as exc:
            tdim2.validate()
        assert "bad-default is not a valid value for this Dimension." in str(exc.value)

        tdim.original_dimension._kwargs.pop('size')
        tdim2.original_dimension._default_value = Dimension.NO_DEFAULT_VALUE

    def test_get_prior_string(self, tdim, tdim2):
        """Apply the transformation on top of the prior string of original dimension."""
        assert tdim.get_prior_string() == "Quantize(norm(0.9, shape=(3, 2)))"
        assert tdim2.get_prior_string() == "OneHotEncode(Enumerate(choices({'asdfa': 0.10, '2': 0.20, '3': 0.30, '4': 0.40})))"  # noqa

    def test_get_string(self, tdim, tdim2):
        """Apply the transformation only on top of the prior string of original dimension."""
        assert tdim.get_string() == "yolo~Quantize(norm(0.9, shape=(3, 2)))"
        assert tdim2.get_string() == "yolo2~OneHotEncode(Enumerate(choices({'asdfa': 0.10, '2': 0.20, '3': 0.30, '4': 0.40})))"  # noqa

    def test_repr(self, tdim):
        """Check method `__repr__`."""
        assert str(tdim) == "Quantize(Real(name=yolo, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None))"  # noqa

    def test_name_property(self, tdim):
        """Check property `name`."""
        assert tdim.name == 'yolo'

    def test_type_property(self, tdim, tdim2):
        """Check property `type`."""
        assert tdim.type == 'integer'
        assert tdim2.type == 'real'

    def test_prior_name_property(self, tdim, tdim2):
        """Check property `prior_name`."""
        assert tdim.prior_name == 'norm'
        assert tdim2.prior_name == 'choices'

    def test_shape_property(self, tdim, tdim2):
        """Check property `shape`."""
        assert tdim.original_dimension.shape == (3, 2)
        assert tdim.shape == (3, 2)
        assert tdim2.original_dimension.shape == ()
        assert tdim2.shape == (4,)

    def test_default_value_property(self, tdim, tdim2):
        """Check property `default_value`."""
        assert tdim.default_value is None
        tdim2.original_dimension._default_value = '3'
        assert numpy.all(tdim2.default_value == (0., 0., 1., 0.))
        tdim2.original_dimension._default_value = None

    def test_cast(self, tdim, tdim2):
        """Check casting through transformers"""
        assert tdim.cast(['10.1']) == [10.0]
        assert numpy.all(tdim2.cast(['asdfa']) == numpy.array([[1, 0, 0, 0]]))
        assert numpy.all(tdim2.cast(['3']) == numpy.array([[0, 0, 1, 0]]))


@pytest.fixture(scope='module')
def space(dim, dim2):
    """Create an example `Space`."""
    space = Space()
    space.register(dim)
    space.register(dim2)
    return space


@pytest.fixture(scope='module')
def tspace(tdim, tdim2):
    """Create an example `TransformedSpace`."""
    tspace = TransformedSpace()
    tspace.register(tdim)
    tspace.register(tdim2)
    return tspace


class TestTransformedSpace(object):
    """Check functionality of class `TransformedSpace`."""

    def test_extends_space(self, tspace):
        """Check that `TransformedSpace` is actually a `Space`."""
        assert isinstance(tspace, Space)

    def test_transform(self, space, tspace, seed):
        """Check method `transform`."""
        yo = space.sample(seed=seed)[0]
        tyo = tspace.transform(yo)
        assert tyo in tspace

    def test_reverse(self, space, tspace, seed):
        """Check method `reverse`."""
        tyo = tspace.sample(seed=seed)[0]
        yo = tspace.reverse(tyo)
        assert yo in space


@pytest.fixture(scope='module')
def space_each_type(dim, dim2):
    """Create an example `Space`."""
    space = Space()
    space.register(dim)
    space.register(dim2)
    space.register(Integer('yolo3', 'randint', 3, 10))
    return space


class TestRequiredSpaceBuilder(object):
    """Check functionality of builder function `build_required_space`."""

    def test_not_supported_requirement(self, space_each_type):
        """Require something which is not supported."""
        with pytest.raises(TypeError) as exc:
            build_required_space('fasdfasf', space_each_type)
        assert 'Unsupported' in str(exc.value)

    def test_no_requirement(self, space_each_type):
        """Check what is built using 'None' requirement."""
        tspace = build_required_space(None, space_each_type)
        assert len(tspace) == 3
        assert tspace[0].type == 'real'
        assert tspace[1].type == 'categorical'
        assert tspace[2].type == 'integer'
        assert (str(tspace) ==
                "Space([Precision(4, Real(name=yolo, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),\n"  # noqa
                "       Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None),\n"  # noqa
                "       Integer(name=yolo3, prior={randint: (3, 10), {}}, shape=(), default value=None)])")  # noqa

        tspace = build_required_space([], space_each_type)
        assert len(tspace) == 3
        assert tspace[0].type == 'real'
        assert tspace[1].type == 'categorical'
        assert tspace[2].type == 'integer'
        assert (str(tspace) ==
                "Space([Precision(4, Real(name=yolo, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),\n"  # noqa
                "       Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None),\n"  # noqa
                "       Integer(name=yolo3, prior={randint: (3, 10), {}}, shape=(), default value=None)])")  # noqa

    def test_integer_requirement(self, space_each_type):
        """Check what is built using 'integer' requirement."""
        tspace = build_required_space('integer', space_each_type)
        assert len(tspace) == 3
        assert tspace[0].type == 'integer'
        assert tspace[1].type == 'integer'
        assert tspace[2].type == 'integer'
        assert(str(tspace) ==
               "Space([Quantize(Real(name=yolo, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),\n"  # noqa
               "       Enumerate(Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None)),\n"  # noqa
               "       Integer(name=yolo3, prior={randint: (3, 10), {}}, shape=(), default value=None)])")  # noqa

    def test_real_requirement(self, space_each_type):
        """Check what is built using 'real' requirement."""
        tspace = build_required_space('real', space_each_type)
        assert len(tspace) == 3
        assert tspace[0].type == 'real'
        assert tspace[1].type == 'real'
        assert tspace[2].type == 'real'
        assert(str(tspace) ==
               "Space([Precision(4, Real(name=yolo, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),\n"  # noqa
               "       OneHotEncode(Enumerate(Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None))),\n"  # noqa
               "       ReverseQuantize(Integer(name=yolo3, prior={randint: (3, 10), {}}, shape=(), default value=None))])")  # noqa

    def test_capacity(self, space_each_type):
        """Check transformer space capacity"""
        tspace = build_required_space('real', space_each_type)
        assert tspace.cardinality == numpy.inf

        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ('asdfa', 2, 3, 4)
        dim = Categorical('yolo', OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer('yolo2', 'uniform', -3, 6)
        space.register(dim)
        tspace = build_required_space('integer', space)
        assert tspace.cardinality == (4 * 2) * 6

        dim = Integer('yolo3', 'uniform', -3, 6, shape=(2, 1))
        space.register(dim)
        tspace = build_required_space('integer', space)
        assert tspace.cardinality == (4 * 2) * 6 * 6 * (2 * 1)


def test_quantization_does_not_violate_bounds():
    """Regress on bug that converts valid float in tdim to non valid excl. upper bound."""
    dim = Integer('yo', 'uniform', 3, 7)
    transformers = [Reverse(Quantize())]
    tdim = TransformedDimension(Compose(transformers, dim.type), dim)
    assert 11 not in dim
    assert 10 in dim
    # but be careful, because upper bound is inclusive
    assert 11.5 not in tdim
    assert 10.6 in tdim
    assert tdim.reverse(9.6) in dim
    # solution is to quantize with 'floor' instead of 'round'
    assert tdim.reverse(9.6) == 9
