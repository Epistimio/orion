#!/usr/bin/env python
"""Test base functionalities of :mod:`orion.core.utils`."""

import pytest

from orion.core.utils import Factory, GenericFactory, float_to_digits_list


def test_deprecated_factory_subclasses_detection():
    """Verify that meta-class Factory finds all subclasses"""
    # TODO: Remove in v0.3.0

    class Base:
        pass

    class A(Base):
        pass

    class B(Base):
        pass

    class AA(A):
        pass

    class AB(A):
        pass

    class AAA(AA):
        pass

    class AA_AB(AA, AB):
        pass

    class MyFactory(Base, metaclass=Factory):
        pass

    assert type(MyFactory(of_type="A")) is A
    assert type(MyFactory(of_type="B")) is B
    assert type(MyFactory(of_type="AA")) is AA
    assert type(MyFactory(of_type="AAA")) is AAA
    assert type(MyFactory(of_type="AA_AB")) is AA_AB

    # Test if there is duplicates
    assert MyFactory.types == {
        cls.__name__.lower(): cls for cls in [A, B, AA, AB, AAA, AA_AB]
    }

    with pytest.raises(NotImplementedError) as exc_info:
        MyFactory(of_type="random")
    assert "Could not find implementation of Base, type = 'random'" in str(
        exc_info.value
    )

    class Random(Base):
        pass

    assert type(MyFactory(of_type="random")) is Random


def test_new_factory_subclasses_detection():
    """Verify that Factory finds all subclasses"""

    class Base:
        pass

    class A(Base):
        pass

    class B(Base):
        pass

    class AA(A):
        pass

    class AB(A):
        pass

    class AAA(AA):
        pass

    class AA_AB(AA, AB):
        pass

    factory = GenericFactory(Base)

    assert type(factory.create(of_type="A")) is A
    assert type(factory.create(of_type="B")) is B
    assert type(factory.create(of_type="AA")) is AA
    assert type(factory.create(of_type="AAA")) is AAA
    assert type(factory.create(of_type="AA_AB")) is AA_AB

    with pytest.raises(NotImplementedError) as exc_info:
        factory.create(of_type="random")
    assert "Could not find implementation of Base, type = 'random'" in str(
        exc_info.value
    )

    class Random(Base):
        pass

    assert type(factory.create(of_type="random")) is Random


@pytest.mark.parametrize(
    "number,digits_list",
    [
        (float("inf"), []),
        (0.0, [0]),
        (0.00001, [1]),
        (12.0, [1, 2]),
        (123000.0, [1, 2, 3]),
        (10.0001, [1, 0, 0, 0, 0, 1]),
        (1e-50, [1]),
        (5.32156e-3, [5, 3, 2, 1, 5, 6]),
    ],
)
def test_float_to_digits_list(number, digits_list):
    """Test that floats are correctly converted to list of digits"""
    assert float_to_digits_list(number) == digits_list
