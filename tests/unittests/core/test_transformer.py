#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.transformer`."""
import copy
import itertools
from collections import OrderedDict

import numpy
import pytest

from orion.algo.space import Categorical, Dimension, Integer, Real, Space
from orion.core.worker.transformer import (
    Compose,
    Enumerate,
    Identity,
    Linearize,
    OneHotEncode,
    Precision,
    Quantize,
    ReshapedDimension,
    ReshapedSpace,
    Reverse,
    TransformedDimension,
    TransformedSpace,
    View,
    build_required_space,
)


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

        t = Identity("mpogias")
        assert t.domain_type == "mpogias"
        assert t.target_type == "mpogias"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Identity()
        assert t.transform("yo") == "yo"

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Identity()
        assert t.reverse("yo") == "yo"

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Identity()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Identity()
        assert t.repr_format("asfa") == "asfa"


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
        assert t.domain_type == "integer"
        assert t.target_type == "real"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Reverse(Quantize())
        assert t.transform(9) == 9.0
        assert t.transform(5) == 5.0
        assert numpy.all(t.transform([9, 5]) == numpy.array([9.0, 5.0], dtype=float))

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
        assert t.repr_format("asfa") == "ReverseQuantize(asfa)"


class TestCompose(object):
    """Test subclasses of `Compose` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Compose([])
        assert t.domain_type is None
        assert t.target_type is None

        t = Compose([], "real")
        assert t.domain_type == "real"
        assert t.target_type == "real"

        t = Compose([Quantize()], "real")
        assert t.domain_type == "real"
        assert t.target_type == "integer"

        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        assert t.domain_type == "categorical"
        assert t.target_type == "real"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        assert numpy.all(t.transform(2) == numpy.array((1.0, 0.0, 0.0)))
        assert numpy.all(t.transform("asfa") == numpy.array((0.0, 1.0, 0.0)))
        assert numpy.all(t.transform("ipsi") == numpy.array((0.0, 0.0, 1.0)))
        with pytest.raises(KeyError):
            t.transform("aafdasfa")
        assert numpy.all(
            t.transform([["ipsi", "asfa"], [2, "ipsi"]])
            == numpy.array(
                [[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]]
            )
        )

        t = Compose([Enumerate([2, "asfa"]), OneHotEncode(2)], "categorical")
        assert t.transform(2) == 0.0
        assert t.transform("asfa") == 1.0
        with pytest.raises(KeyError):
            t.transform("ipsi")
        assert numpy.all(
            t.transform([["asfa", "asfa"], [2, "asfa"]])
            == numpy.array([[1.0, 1.0], [0.0, 1.0]])
        )

        # for the crazy enough
        t = Compose([Enumerate([2]), OneHotEncode(1)], "categorical")
        assert t.transform(2) == 0.0
        with pytest.raises(KeyError):
            t.transform("ipsi")
        assert numpy.all(t.transform([[2, 2], [2, 2]]) == [[0, 0], [0, 0]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        assert t.reverse((0.9, 0.8, 0.3)) == 2
        assert t.reverse((-0.3, 2.0, 0.0)) == "asfa"
        assert t.reverse((0.0, 0.0, 1.0)) == "ipsi"
        with pytest.raises(AssertionError):
            t.reverse((0.0, 0.0, 0.0, 1.0))
        assert numpy.all(
            t.reverse(
                numpy.array(
                    [
                        [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
                        [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
                    ]
                )
            )
            == numpy.array([["ipsi", "asfa"], [2, "ipsi"]], dtype=numpy.object)
        )

        t = Compose([Enumerate([2, "asfa"]), OneHotEncode(2)], "categorical")
        assert t.reverse(0.3) == 2
        assert t.reverse(2.0) == "asfa"
        assert numpy.all(
            t.reverse((0.0, 0.0, 0.0, 1.0))
            == numpy.array([2, 2, 2, "asfa"], dtype=numpy.object)
        )
        assert numpy.all(
            t.reverse(numpy.array([[0.55, 3.0], [-0.6, 1.0]]))
            == numpy.array([["asfa", "asfa"], [2, "asfa"]], dtype=numpy.object)
        )

        # for the crazy enough
        t = Compose([Enumerate([2]), OneHotEncode(1)], "categorical")
        assert t.reverse(0) == 2
        assert t.reverse(5.0) == 2
        assert t.reverse(0.2) == 2
        assert t.reverse(-0.2) == 2
        assert numpy.all(
            t.reverse([[0.5, 0], [1.0, 55]])
            == numpy.array([[2, 2], [2, 2]], dtype=numpy.object)
        )

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        assert t.infer_target_shape((2, 5)) == (2, 5, 3)

        t = Compose([Enumerate([2, "asfa"]), OneHotEncode(2)], "categorical")
        assert t.infer_target_shape((2, 5)) == (2, 5)

        t = Compose([Enumerate([2]), OneHotEncode(1)], "categorical")
        assert t.infer_target_shape((2, 5)) == (2, 5)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Compose([Enumerate([2, "asfa", "ipsi"]), OneHotEncode(3)], "categorical")
        assert t.repr_format("asfa") == "OneHotEncode(Enumerate(asfa))"


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
        assert t.domain_type == "real"
        assert t.target_type == "real"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Precision(precision=4)
        assert t.transform(8.654321098) == 8.654
        assert t.transform(0.000123456789) == 0.0001235
        assert numpy.all(
            t.transform([8.654321098, 0.000123456789])
            == numpy.array([8.654, 0.0001235], dtype=float)
        )

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Precision()
        assert t.reverse(9.0) == 9.0
        assert t.reverse(5.0) == 5.0
        assert numpy.all(t.reverse([9.0, 5.0]) == numpy.array([9.0, 5.0], dtype=float))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Precision()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Precision()
        assert t.repr_format("asfa") == "Precision(4, asfa)"


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
        assert t.domain_type == "real"
        assert t.target_type == "integer"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Quantize()
        assert t.transform(8.6) == 8
        assert t.transform(5.3) == 5
        assert numpy.all(t.transform([8.6, 5.3]) == numpy.array([8, 5], dtype=int))

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Quantize()
        assert t.reverse(9) == 9.0
        assert t.reverse(5) == 5.0
        assert numpy.all(t.reverse([9, 5]) == numpy.array([9.0, 5.0], dtype=float))

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Quantize()
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Quantize()
        assert t.repr_format("asfa") == "Quantize(asfa)"


class TestEnumerate(object):
    """Test subclasses of `Enumerate` transformation."""

    def test_deepcopy(self):
        """Verify that the transformation object can be copied"""
        t = Enumerate([2, "asfa", "ipsi"])
        # Copy won't fail if vectorized function is not called at least once.
        t.transform([2])
        copy.deepcopy(t)

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Enumerate([2, "asfa", "ipsi"])
        assert t.domain_type == "categorical"
        assert t.target_type == "integer"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Enumerate([2, "asfa", "ipsi"])
        assert t.transform(2) == 0
        assert t.transform("asfa") == 1
        assert t.transform("ipsi") == 2
        with pytest.raises(KeyError):
            t.transform("aafdasfa")
        assert numpy.all(
            t.transform([["ipsi", "asfa"], [2, "ipsi"]]) == [[2, 1], [0, 2]]
        )

        # for the crazy enough
        t = Enumerate([2])
        assert t.transform(2) == 0
        with pytest.raises(KeyError):
            t.transform("aafdasfa")
        assert numpy.all(t.transform([[2, 2], [2, 2]]) == [[0, 0], [0, 0]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = Enumerate([2, "asfa", "ipsi"])
        assert t.reverse(0) == 2
        assert t.reverse(1) == "asfa"
        assert t.reverse(2) == "ipsi"
        with pytest.raises(IndexError):
            t.reverse(3)
        assert numpy.all(
            t.reverse([[2, 1], [0, 2]])
            == numpy.array([["ipsi", "asfa"], [2, "ipsi"]], dtype=numpy.object)
        )

        # for the crazy enough
        t = Enumerate([2])
        assert t.reverse(0) == 2
        with pytest.raises(IndexError):
            t.reverse(1)
        assert numpy.all(
            t.reverse([[0, 0], [0, 0]])
            == numpy.array([[2, 2], [2, 2]], dtype=numpy.object)
        )

    def test_infer_target_shape(self):
        """Check if it infers the shape of a transformed `Dimension`."""
        t = Enumerate([2, "asfa", "ipsi"])
        assert t.infer_target_shape((5,)) == (5,)

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Enumerate([2, "asfa", "ipsi"])
        assert t.repr_format("asfa") == "Enumerate(asfa)"


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
        assert t.domain_type == "integer"
        assert t.target_type == "real"

    def test_transform(self):
        """Check if it transforms properly."""
        t = OneHotEncode(3)
        assert numpy.all(t.transform(0) == numpy.array((1.0, 0.0, 0.0)))
        assert numpy.all(t.transform(1) == numpy.array((0.0, 1.0, 0.0)))
        assert numpy.all(t.transform(2) == numpy.array((0.0, 0.0, 1.0)))
        with pytest.raises(AssertionError):
            t.transform(4)
        with pytest.raises(AssertionError):
            t.transform(-1)
        with pytest.raises(AssertionError):
            t.transform(2.2)
        assert numpy.all(
            t.transform([[2, 1], [0, 2]])
            == numpy.array(
                [[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]]
            )
        )

        t = OneHotEncode(2)
        assert t.transform(0) == 0.0
        assert t.transform(1) == 1.0
        with pytest.raises(TypeError):
            t.transform("ipsi")
        assert numpy.all(
            t.transform([[1, 1], [0, 1]]) == numpy.array([[1.0, 1.0], [0.0, 1.0]])
        )

        # for the crazy enough
        t = OneHotEncode(1)
        assert t.transform(0) == 0.0
        with pytest.raises(TypeError):
            t.transform("ipsi")
        assert numpy.all(t.transform([[0, 0], [0, 0]]) == [[0.0, 0.0], [0.0, 0.0]])

    def test_reverse(self):
        """Check if it reverses `transform` properly, if possible."""
        t = OneHotEncode(3)
        assert t.reverse((0.9, 0.8, 0.3)) == 0
        assert t.reverse((-0.3, 2.0, 0.0)) == 1
        assert t.reverse((0.0, 0.0, 1.0)) == 2
        with pytest.raises(AssertionError):
            t.reverse((0.0, 0.0, 0.0, 1.0))
        assert numpy.all(
            t.reverse(
                numpy.array(
                    [
                        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    ]
                )
            )
            == numpy.array([[2, 1], [0, 2]], dtype=int)
        )

        t = OneHotEncode(2)
        assert t.reverse(0.3) == 0
        assert t.reverse(2.0) == 1
        assert numpy.all(
            t.reverse((0.0, 0.0, 0.0, 1.0)) == numpy.array([0, 0, 0, 1], dtype=int)
        )
        assert numpy.all(
            t.reverse(numpy.array([[0.55, 3.0], [-0.6, 1.0]]))
            == numpy.array([[1, 1], [0, 1]], dtype=int)
        )

        # for the crazy enough
        t = OneHotEncode(1)
        assert t.reverse(0) == 0
        assert t.reverse(5.0) == 0
        assert t.reverse(0.2) == 0
        assert t.reverse(-0.2) == 0
        assert numpy.all(
            t.reverse([[0.5, 0], [1.0, 55]]) == numpy.array([[0, 0], [0, 0]], dtype=int)
        )

    def test_interval(self):
        """Test that the onehot interval has the proper dimensions"""
        t = OneHotEncode(3)
        low, high = t.interval()
        assert (low == numpy.zeros(3)).all()
        assert (high == numpy.ones(3)).all()

        t = OneHotEncode(2)
        low, high = t.interval()
        assert (low == numpy.zeros(1)).all()
        assert (high == numpy.ones(1)).all()

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
        assert t.repr_format("asfa") == "OneHotEncode(asfa)"


class TestLinearize(object):
    """Test subclasses of `Linearize` transformation."""

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do
        what's expected.
        """
        t = Linearize()
        assert t.domain_type == "real"
        assert t.target_type == "real"

    def test_transform(self):
        """Check if it transforms properly."""
        t = Linearize()
        assert t.transform(numpy.e) == 1
        t.transform(0)

    def test_reverse(self):
        """Check if it reverses `transform` properly."""
        t = Linearize()
        assert t.reverse(1) == numpy.e

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        t = Linearize()
        assert t.repr_format(1.0) == "Linearize(1.0)"


class TestView(object):
    """Test subclasses of `View` transformation."""

    def test_domain_and_target_type(self):
        """Check if attribute-like `domain_type` and `target_type` do what's expected."""
        t = View(shape=None, index=None, domain_type="some fancy type")
        assert t.domain_type == "some fancy type"
        assert t.target_type == "some fancy type"

    def test_transform(self):
        """Check if it transforms properly."""
        shape = (3, 4, 5)
        index = (0, 2, 1)
        t = View(shape=shape, index=index)
        a = numpy.zeros(shape)
        a[index] = 2
        assert t.transform(a) == 2

    def test_reverse(self):
        """Check if it reverses `transform` properly."""
        shape = (3, 4, 5)
        index = (0, 2, 1)
        a = numpy.zeros(shape)
        a[index] = 2
        flattened = a.reshape(-1).tolist()
        point = [None] + flattened + [None]
        t = View(shape=shape, index=(0, 0, 0))
        numpy.testing.assert_equal(t.reverse(point, 1), a)

    def test_first(self):
        """Test that views are correctly identified as first"""
        shape = (3, 4, 5)
        assert View(shape=shape, index=(0, 0, 0)).first
        assert not View(shape=shape, index=(0, 1, 0)).first

    def test_repr_format(self):
        """Check representation of a transformed dimension."""
        shape = (3, 4, 5)
        index = (0, 2, 1)
        t = View(shape=shape, index=index)
        assert t.repr_format(1.0) == "View(shape=(3, 4, 5), index=(0, 2, 1), 1.0)"


@pytest.fixture(scope="module")
def dim():
    """Create an example of `Dimension`."""
    dim = Real("yolo0", "norm", 0.9, shape=(3, 2))
    return dim


@pytest.fixture(scope="module")
def logdim():
    """Create an log example of `Dimension`."""
    dim = Real("yolo4", "reciprocal", 1.0, 10.0, shape=(3, 2))
    return dim


@pytest.fixture(scope="module")
def logintdim():
    """Create an log integer example of `Dimension`."""
    dim = Integer("yolo5", "reciprocal", 1, 10, shape=(3, 2))
    return dim


@pytest.fixture(scope="module")
def tdim(dim):
    """Create an example of `TransformedDimension`."""
    transformers = [Quantize()]
    tdim = TransformedDimension(Compose(transformers, dim.type), dim)
    return tdim


@pytest.fixture(scope="module")
def rdims(tdim):
    """Create an example of `ReshapedDimension`."""
    transformations = {}
    for index in itertools.product(*map(range, tdim.shape)):
        key = f'{tdim.name}[{",".join(map(str, index))}]'
        transformations[key] = ReshapedDimension(
            transformer=View(tdim.shape, index, tdim.type),
            original_dimension=tdim,
            name=key,
            index=0,
        )

    return transformations


@pytest.fixture(scope="module")
def rdim(dim, rdims):
    """Single ReshapedDimension"""
    return rdims[f"{dim.name}[0,1]"]


@pytest.fixture(scope="module")
def dim2():
    """Create a second example of `Dimension`."""
    probs = (0.1, 0.2, 0.3, 0.4)
    categories = ("asdfa", "2", "3", "4")
    categories = OrderedDict(zip(categories, probs))
    dim2 = Categorical("yolo2", categories)
    return dim2


@pytest.fixture(scope="module")
def tdim2(dim2):
    """Create a second example of `TransformedDimension`."""
    transformers = [Enumerate(dim2.categories), OneHotEncode(len(dim2.categories))]
    tdim2 = TransformedDimension(Compose(transformers, dim2.type), dim2)
    return tdim2


@pytest.fixture(scope="module")
def rdims2(tdim2):
    """Create a categorical example of `ReshapedDimension`."""
    transformations = {}
    for index in itertools.product(*map(range, tdim2.shape)):
        key = f'{tdim2.name}[{",".join(map(str, index))}]'
        transformations[key] = ReshapedDimension(
            transformer=View(tdim2.shape, index, tdim2.type),
            original_dimension=tdim2,
            name=key,
            index=1,
        )

    return transformations


@pytest.fixture(scope="module")
def rdim2(dim2, rdims2):
    """Single ReshapedDimension"""
    return rdims2[f"{dim2.name}[1]"]


@pytest.fixture(scope="module")
def dim3():
    """Create an example of integer `Dimension`."""
    return Integer("yolo3", "uniform", 3, 7)


@pytest.fixture(scope="module")
def tdim3(dim3):
    """Create an example of integer `Dimension`."""
    return TransformedDimension(Compose([], dim3.type), dim3)


@pytest.fixture(scope="module")
def rdims3(tdim3):
    """Create an example of integer `Dimension`."""
    rdim3 = ReshapedDimension(
        transformer=Identity(tdim3.type), original_dimension=tdim3, index=2
    )

    return {tdim3.name: rdim3}


class TestTransformedDimension(object):
    """Check functionality of class `TransformedDimension`."""

    def test_transform(self, tdim):
        """Check method `transform`."""
        assert tdim.transform(8.6) == 8
        assert tdim.transform(5.3) == 5
        assert numpy.all(tdim.transform([8.6, 5.3]) == numpy.array([8, 5], dtype=int))

    def test_reverse(self, tdim):
        """Check method `reverse`."""
        assert tdim.reverse(9) == 9.0
        assert tdim.reverse(5) == 5.0
        assert numpy.all(tdim.reverse([9, 5]) == numpy.array([9.0, 5.0], dtype=float))

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
        low, high = tdim2.interval()
        assert (low == numpy.zeros(4)).all()
        assert (high == numpy.ones(4)).all()

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
        assert tdim._get_hashable_members() == (
            "Compose",
            "Quantize",
            "real",
            "integer",
            "Identity",
            "real",
            "real",
            "yolo0",
            (3, 2),
            "real",
            (0.9,),
            (),
            None,
            "norm",
        )
        assert tdim2._get_hashable_members() == (
            "Compose",
            "OneHotEncode",
            "integer",
            "real",
            4,
            "Compose",
            "Enumerate",
            "categorical",
            "integer",
            "Identity",
            "categorical",
            "categorical",
            "yolo2",
            (),
            "categorical",
            (),
            (),
            None,
            "Distribution",
        )

    def test_validate(self, tdim, tdim2):
        """Validate original_dimension"""
        # It pass
        tdim.validate()
        tdim2.validate()

        # We break it
        tdim.original_dimension._kwargs["size"] = (2,)
        tdim2.original_dimension._default_value = "bad-default"

        # It does not pass
        with pytest.raises(ValueError) as exc:
            tdim.validate()
        assert "Use 'shape' keyword only instead of 'size'." in str(exc.value)

        with pytest.raises(ValueError) as exc:
            tdim2.validate()
        assert "bad-default is not a valid value for this Dimension." in str(exc.value)

        tdim.original_dimension._kwargs.pop("size")
        tdim2.original_dimension._default_value = Dimension.NO_DEFAULT_VALUE

    def test_repr(self, tdim):
        """Check method `__repr__`."""
        assert (
            str(tdim)
            == "Quantize(Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None))"
        )  # noqa

    def test_name_property(self, tdim):
        """Check property `name`."""
        assert tdim.name == "yolo0"

    def test_type_property(self, tdim, tdim2):
        """Check property `type`."""
        assert tdim.type == "integer"
        assert tdim2.type == "real"

    def test_prior_name_property(self, tdim, tdim2):
        """Check property `prior_name`."""
        assert tdim.prior_name == "norm"
        assert tdim2.prior_name == "choices"

    def test_shape_property(self, tdim, tdim2):
        """Check property `shape`."""
        assert tdim.original_dimension.shape == (3, 2)
        assert tdim.shape == (3, 2)
        assert tdim2.original_dimension.shape == ()
        assert tdim2.shape == (4,)


class TestReshapedDimension(object):
    """Check functionality of class `ReshapedDimension`."""

    def test_transform(self, rdim):
        """Check method `transform`."""
        a = numpy.zeros((3, 2))
        a[0, 1] = 2
        assert rdim.transform([a, None]) == 2

    def test_reverse(self, rdim):
        """Check method `reverse`."""
        a = numpy.zeros((3, 2))
        a[0, 1] = 2
        p = a.reshape(-1).tolist() + [None]
        numpy.testing.assert_equal(rdim.reverse(p, 0), a)

    def test_interval(self, rdim):
        """Check method `interval`."""
        assert rdim.interval() == (
            -numpy.array(numpy.inf).astype(int) + 1,
            numpy.array(numpy.inf).astype(int) - 1,
        )

    def test_interval_from_categorical(self, rdim2):
        """Check how we should treat interval when original dimension is categorical."""
        assert rdim2.interval() == (0, 1)

    def test_eq(self, rdim, rdim2):
        """Return True if other is the same transformed dimension as self"""
        assert rdim != rdim2
        assert rdim == copy.deepcopy(rdim)

    def test_hash(self, rdim, rdim2):
        """Test that hash is consistent for identical and different transformed dimensions"""
        assert hash(rdim) != hash(rdim2)
        assert hash(rdim) == hash(copy.deepcopy(rdim))

    def test_get_hashable_members(self, rdim, rdim2):
        """Test that hashable members of the transformed dimensions are the aggregation of
        transformer's and original dimension's hashable members.
        """
        assert rdim._get_hashable_members() == (
            "View",
            "integer",
            "integer",
            "Compose",
            "Quantize",
            "real",
            "integer",
            "Identity",
            "real",
            "real",
            "yolo0",
            (3, 2),
            "real",
            (0.9,),
            (),
            None,
            "norm",
        )
        assert rdim2._get_hashable_members() == (
            "View",
            "real",
            "real",
            "Compose",
            "OneHotEncode",
            "integer",
            "real",
            4,
            "Compose",
            "Enumerate",
            "categorical",
            "integer",
            "Identity",
            "categorical",
            "categorical",
            "yolo2",
            (),
            "categorical",
            (),
            (),
            None,
            "Distribution",
        )

    def test_repr(self, rdim):
        """Check method `__repr__`."""
        assert (
            str(rdim)
            == "View(shape=(3, 2), index=(0, 1), Quantize(Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)))"
        )  # noqa

    def test_name_property(self, rdim):
        """Check property `name`."""
        assert rdim.name == "yolo0[0,1]"

    def test_type_property(self, rdim, rdim2):
        """Check property `type`."""
        assert rdim.type == "integer"
        assert rdim2.type == "real"

    def test_prior_name_property(self, rdim, rdim2):
        """Check property `prior_name`."""
        assert rdim.prior_name == "norm"
        assert rdim2.prior_name == "choices"

    def test_shape_property(self, rdim, rdim2):
        """Check property `shape`."""
        assert rdim.original_dimension.shape == (3, 2)
        assert rdim.shape == ()
        assert rdim2.original_dimension.shape == (4,)
        assert rdim2.shape == ()


@pytest.fixture(scope="module")
def space(dim, dim2, dim3):
    """Create an example `Space`."""
    space = Space()
    space.register(dim)
    space.register(dim2)
    space.register(dim3)
    return space


@pytest.fixture(scope="module")
def tspace(space, tdim, tdim2, tdim3):
    """Create an example `TransformedSpace`."""
    tspace = TransformedSpace(space)
    tspace.register(tdim)
    tspace.register(tdim2)
    tspace.register(tdim3)
    return tspace


@pytest.fixture(scope="module")
def rspace(tspace, rdims, rdims2, rdims3):
    """Create an example `ReshapedSpace`."""
    rspace = ReshapedSpace(tspace)
    for dim in itertools.chain(rdims.values(), rdims2.values(), rdims3.values()):
        rspace.register(dim)

    return rspace


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

    def test_sample(self, space, tspace, seed):
        """Check method `sample`."""
        points = tspace.sample(n_samples=2, seed=seed)
        # pytest.set_trace()
        assert len(points) == 2
        assert points[0] in tspace
        assert points[1] in tspace
        assert tspace.reverse(points[0]) in space
        assert tspace.reverse(points[1]) in space


class TestReshapedSpace(object):
    """Check functionality of class `ReshapeSpace`."""

    def test_reverse(self, space, tspace, rspace, seed):
        """Check method `reverse`."""
        ryo = (
            numpy.zeros(tspace["yolo0"].shape).reshape(-1).tolist()
            + numpy.zeros(tspace["yolo2"].shape).reshape(-1).tolist()
            + [10]
        )
        yo = rspace.reverse(ryo)
        assert yo in space

    def test_contains(self, tspace, rspace, seed):
        """Check method `transform`."""
        ryo = (
            numpy.zeros(tspace["yolo0"].shape).reshape(-1).tolist()
            + numpy.zeros(tspace["yolo2"].shape).reshape(-1).tolist()
            + [10]
        )

        assert ryo in rspace

    def test_transform(self, space, rspace, seed):
        """Check method `transform`."""
        yo = space.sample(seed=seed)[0]
        tyo = rspace.transform(yo)
        assert tyo in rspace

    def test_sample(self, space, rspace, seed):
        """Check method `sample`."""
        points = rspace.sample(n_samples=2, seed=seed)
        assert len(points) == 2
        assert points[0] in rspace
        assert points[1] in rspace
        assert rspace.reverse(points[0]) in space
        assert rspace.reverse(points[1]) in space

    def test_interval(self, rspace):
        """Check method `interval`."""
        interval = rspace.interval()
        assert len(interval) == 3 * 2 + 4 + 1
        for i in range(3 * 2):
            # assert interval[i] == (-float('inf'), float('inf'))
            assert interval[i] == (
                -numpy.array(numpy.inf).astype(int) + 1,
                numpy.array(numpy.inf).astype(int) - 1,
            )
        for i in range(3 * 2, 3 * 2 + 4):
            assert interval[i] == (0, 1)
        assert interval[-1] == (3, 10)

    def test_reshape(self, rspace):
        """Verify that the dimension are reshaped properly, forward and backward"""
        point = [numpy.arange(6).reshape(3, 2), "3", 10]
        rpoint = point[0].reshape(-1).tolist() + [0.0, 0.0, 1.0, 0.0] + [10]
        assert rspace.transform(point) == tuple(rpoint)
        numpy.testing.assert_equal(rspace.reverse(rpoint)[0], point[0])
        assert rspace.reverse(rpoint)[1] == point[1]
        assert rspace.reverse(rpoint)[2] == point[2]

    def test_cardinality(self, dim2):
        """Check cardinality of reshaped space"""
        space = Space()
        space.register(Real("yolo0", "reciprocal", 0.1, 1, precision=1, shape=(2, 2)))
        space.register(dim2)

        rspace = build_required_space(space, shape_requirement="flattened")
        assert rspace.cardinality == (10 ** (2 * 2)) * 4

        space = Space()
        space.register(Real("yolo0", "uniform", 0, 2, shape=(2, 2)))
        space.register(dim2)

        rspace = build_required_space(
            space, type_requirement="integer", shape_requirement="flattened"
        )
        assert rspace.cardinality == (3 ** (2 * 2)) * 4


@pytest.fixture(scope="module")
def space_each_type(dim, dim2, dim3, logdim, logintdim):
    """Create an example `Space`."""
    space = Space()
    space.register(dim)
    space.register(dim2)
    space.register(dim3)
    space.register(logdim)
    space.register(logintdim)
    return space


class TestRequiredSpaceBuilder(object):
    """Check functionality of builder function `build_required_space`."""

    @pytest.mark.xfail(
        reason="Bring it back when testing new builder and extend to shape and dist"
    )
    def test_not_supported_requirement(self, space_each_type):
        """Require something which is not supported."""
        with pytest.raises(TypeError) as exc:
            build_required_space(space_each_type, type_requirement="fasdfasf")
        assert "Unsupported" in str(exc.value)

    def test_no_requirement(self, space_each_type):
        """Check what is built using 'None' requirement."""
        tspace = build_required_space(space_each_type)
        assert len(tspace) == 5
        assert tspace[0].type == "real"
        assert tspace[1].type == "categorical"
        # NOTE:HEAD
        assert tspace[2].type == "integer"
        assert tspace[3].type == "real"
        assert tspace[4].type == "integer"
        assert (
            str(tspace)
            == """\
Space([Precision(4, Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),
       Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None),
       Integer(name=yolo3, prior={uniform: (3, 7), {}}, shape=(), default value=None),
       Precision(4, Real(name=yolo4, prior={reciprocal: (1.0, 10.0), {}}, shape=(3, 2), default value=None)),
       Integer(name=yolo5, prior={reciprocal: (1, 10), {}}, shape=(3, 2), default value=None)])\
"""
        )  # noqa

    def test_integer_requirement(self, space_each_type):
        """Check what is built using 'integer' requirement."""
        tspace = build_required_space(space_each_type, type_requirement="integer")
        assert len(tspace) == 5
        assert tspace[0].type == "integer"
        assert tspace[1].type == "integer"
        assert tspace[2].type == "integer"
        assert tspace[3].type == "integer"
        assert tspace[4].type == "integer"
        assert (
            str(tspace)
            == """\
Space([Quantize(Precision(4, Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None))),
       Enumerate(Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None)),
       Integer(name=yolo3, prior={uniform: (3, 7), {}}, shape=(), default value=None),
       Quantize(Precision(4, Real(name=yolo4, prior={reciprocal: (1.0, 10.0), {}}, shape=(3, 2), default value=None))),
       Integer(name=yolo5, prior={reciprocal: (1, 10), {}}, shape=(3, 2), default value=None)])\
"""
        )  # noqa

    def test_real_requirement(self, space_each_type):
        """Check what is built using 'real' requirement."""
        tspace = build_required_space(space_each_type, type_requirement="real")
        assert len(tspace) == 5
        assert tspace[0].type == "real"
        assert tspace[1].type == "real"
        assert tspace[2].type == "real"
        assert tspace[3].type == "real"
        assert tspace[4].type == "real"
        assert (
            str(tspace)
            == """\
Space([Precision(4, Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),
       OneHotEncode(Enumerate(Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None))),
       ReverseQuantize(Integer(name=yolo3, prior={uniform: (3, 7), {}}, shape=(), default value=None)),
       Precision(4, Real(name=yolo4, prior={reciprocal: (1.0, 10.0), {}}, shape=(3, 2), default value=None)),
       ReverseQuantize(Integer(name=yolo5, prior={reciprocal: (1, 10), {}}, shape=(3, 2), default value=None))])\
"""
        )  # noqa

    def test_numerical_requirement(self, space_each_type):
        """Check what is built using 'integer' requirement."""
        tspace = build_required_space(space_each_type, type_requirement="numerical")
        assert len(tspace) == 5
        assert tspace[0].type == "real"
        assert tspace[1].type == "integer"
        assert tspace[2].type == "integer"
        assert tspace[3].type == "real"
        assert tspace[4].type == "integer"
        assert (
            str(tspace)
            == """\
Space([Precision(4, Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),
       Enumerate(Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None)),
       Integer(name=yolo3, prior={uniform: (3, 7), {}}, shape=(), default value=None),
       Precision(4, Real(name=yolo4, prior={reciprocal: (1.0, 10.0), {}}, shape=(3, 2), default value=None)),
       Integer(name=yolo5, prior={reciprocal: (1, 10), {}}, shape=(3, 2), default value=None)])\
"""
        )  # noqa

    def test_linear_requirement(self, space_each_type):
        """Check what is built using 'linear' requirement."""
        tspace = build_required_space(space_each_type, dist_requirement="linear")
        assert len(tspace) == 5
        assert tspace[0].type == "real"
        assert tspace[1].type == "categorical"
        assert tspace[2].type == "integer"
        assert tspace[3].type == "real"
        assert tspace[4].type == "integer"
        assert (
            str(tspace)
            == """\
Space([Precision(4, Real(name=yolo0, prior={norm: (0.9,), {}}, shape=(3, 2), default value=None)),
       Categorical(name=yolo2, prior={asdfa: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}, shape=(), default value=None),
       Integer(name=yolo3, prior={uniform: (3, 7), {}}, shape=(), default value=None),
       Linearize(Precision(4, Real(name=yolo4, prior={reciprocal: (1.0, 10.0), {}}, shape=(3, 2), default value=None))),
       Quantize(Linearize(ReverseQuantize(Integer(name=yolo5, prior={reciprocal: (1, 10), {}}, shape=(3, 2), default value=None))))])\
"""
        )  # noqa

    def test_flatten_requirement(self, space_each_type):
        """Check what is built using 'flatten' requirement."""
        tspace = build_required_space(space_each_type, shape_requirement="flattened")

        # 1 integer + 1 categorical + 1 * (3, 2) shapes
        assert len(tspace) == 1 + 1 + 3 * (3 * 2)
        assert str(tspace).count("View") == 3 * (3 * 2)

        i = 0
        for _ in range(3 * 2):
            assert tspace[i].type == "real"
            i += 1

        assert tspace[i].type == "categorical"
        i += 1

        assert tspace[i].type == "integer"
        i += 1

        for _ in range(3 * 2):
            assert tspace[i].type == "real"
            i += 1

        for _ in range(3 * 2):
            assert tspace[i].type == "integer"
            i += 1

        tspace = build_required_space(
            space_each_type, shape_requirement="flattened", type_requirement="real"
        )

        # 1 integer + 4 categorical + 1 * (3, 2) shapes
        assert len(tspace) == 1 + 4 + 3 * (3 * 2)
        assert str(tspace).count("View") == 4 + 3 * (3 * 2)

    def test_capacity(self, space_each_type):
        """Check transformer space capacity"""
        tspace = build_required_space(space_each_type, type_requirement="real")
        assert tspace.cardinality == numpy.inf

        space = Space()
        probs = (0.1, 0.2, 0.3, 0.4)
        categories = ("asdfa", 2, 3, 4)
        dim = Categorical("yolo0", OrderedDict(zip(categories, probs)), shape=2)
        space.register(dim)
        dim = Integer("yolo2", "uniform", -3, 6)
        space.register(dim)
        tspace = build_required_space(space, type_requirement="integer")
        assert tspace.cardinality == (4 ** 2) * (6 + 1)

        dim = Integer("yolo3", "uniform", -3, 6, shape=(2, 1))
        space.register(dim)
        tspace = build_required_space(space, type_requirement="integer")
        assert tspace.cardinality == (4 ** 2) * (6 + 1) * ((6 + 1) ** (2 * 1))

        tspace = build_required_space(
            space, type_requirement="integer", shape_requirement="flattened"
        )
        assert tspace.cardinality == (4 ** 2) * (6 + 1) * ((6 + 1) ** (2 * 1))

        tspace = build_required_space(
            space, type_requirement="integer", dist_requirement="linear"
        )
        assert tspace.cardinality == (4 ** 2) * (6 + 1) * ((6 + 1) ** (2 * 1))


def test_quantization_does_not_violate_bounds():
    """Regress on bug that converts valid float in tdim to non valid excl. upper bound."""
    dim = Integer("yo", "uniform", 3, 7)
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


def test_precision_with_linear(space, logdim, logintdim):
    """Test that precision isn't messed up by linearization."""
    space.register(logdim)
    space.register(logintdim)

    # Force precision on all real or linearized dimensions
    space["yolo0"].precision = 3
    space["yolo4"].precision = 4
    space["yolo5"].precision = 5

    # Create a point
    point = list(space.sample(1)[0])
    real_index = list(space.keys()).index("yolo0")
    logreal_index = list(space.keys()).index("yolo4")
    logint_index = list(space.keys()).index("yolo5")
    point[real_index] = 0.133333
    point[logreal_index] = 0.1222222
    point[logint_index] = 2

    # Check first without linearization
    tspace = build_required_space(space, type_requirement="numerical")
    # Check that transform is fine
    tpoint = tspace.transform(point)
    assert tpoint[real_index] == 0.133
    assert tpoint[logreal_index] == 0.1222
    assert tpoint[logint_index] == 2

    # Check that reserve does not break precision
    rpoint = tspace.reverse(tpoint)
    assert rpoint[real_index] == 0.133
    assert rpoint[logreal_index] == 0.1222
    assert rpoint[logint_index] == 2

    # Check with linearization
    tspace = build_required_space(
        space, dist_requirement="linear", type_requirement="real"
    )
    # Check that transform is fine
    tpoint = tspace.transform(point)
    assert tpoint[real_index] == 0.133
    assert tpoint[logreal_index] == numpy.log(0.1222)
    assert tpoint[logint_index] == numpy.log(2)

    # Check that reserve does not break precision
    rpoint = tspace.reverse(tpoint)
    assert rpoint[real_index] == 0.133
    assert rpoint[logreal_index] == 0.1222
    assert rpoint[logint_index] == 2
